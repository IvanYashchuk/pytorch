#include <c10/util/Exception.h>
#include <torch/csrc/dynamo/extra_state.h>

#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/debug_macros.h>
#include <torch/csrc/dynamo/framelocals_mapping.h>
#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/utils/python_compat.h>

#if IS_PYTHON_3_12_PLUS
#define _PyCode_GetExtra PyUnstable_Code_GetExtra
#define _PyCode_SetExtra PyUnstable_Code_SetExtra
#endif

namespace {
// Short-term fix for: https://github.com/pytorch/pytorch/issues/166926
bool use_lru = true;
} // namespace

Py_ssize_t extra_index = -1;

CacheEntry* ExtraState::get_first_entry() {
  if (this->cache_entry_list.empty()) {
    return nullptr;
  }
  return &this->cache_entry_list.front();
}

ExtraState::ExtraState(PyCodeObject* orig_code_arg)
    : orig_code(orig_code_arg) {}

void ExtraState::move_to_front(CacheEntry* cache_entry) {
  CHECK(cache_entry->_owner == this);
  CHECK(!this->cache_entry_list.empty());
  CHECK(cache_entry == &*cache_entry->_owner_loc);
  this->cache_entry_list.splice(
      this->cache_entry_list.begin(),
      this->cache_entry_list,
      cache_entry->_owner_loc);
}

void ExtraState::move_to_back(CacheEntry* cache_entry) {
  CHECK(cache_entry->_owner == this);
  CHECK(!this->cache_entry_list.empty());
  CHECK(cache_entry == &*cache_entry->_owner_loc);
  this->cache_entry_list.splice(
      this->cache_entry_list.end(),
      this->cache_entry_list,
      cache_entry->_owner_loc);
}

void ExtraState::invalidate(
    CacheEntry* cache_entry,
    py::object deleted_guard_manager) {
  // Sometimes setting the cache_entry->code to None causes the orig_code to be
  // freed. This calls destroy_extra_state, which deletes the extra_state and
  // all the cache_entries. This causes the `this` pointer to be a dangling
  // pointer, causing a segfault. So, we manually inc/dec ref the original code
  // pointer to prevent triggering of destroy_extra_state while the invalidate
  // function is running.
  Py_INCREF(this->orig_code);

  CHECK(cache_entry->_owner == this);
  CHECK(!this->cache_entry_list.empty());
  CHECK(cache_entry == &*cache_entry->_owner_loc);
  cache_entry->invalidate(std::move(deleted_guard_manager));
  // Move the cache entry to the end of the list because these will always
  // return False.
  cache_entry->_owner->move_to_back(cache_entry);
  Py_DECREF(this->orig_code);
}

CacheEntry* extract_cache_entry(ExtraState* extra_state) {
  if (extra_state == nullptr) {
    return nullptr;
  }
  return extra_state->get_first_entry();
}

FrameState* extract_frame_state(ExtraState* extra_state) {
  if (extra_state == nullptr) {
    return nullptr;
  }
  return (FrameState*)extra_state->frame_state.ptr();
}

FrameExecStrategy extra_state_get_exec_strategy(ExtraState* extra_state) {
  return extra_state->strategy;
}

void extra_state_set_exec_strategy(
    ExtraState* extra_state,
    FrameExecStrategy strategy) {
  extra_state->strategy = strategy;
}

ExtraState* get_extra_state(PyCodeObject* code) {
  ExtraState* extra = nullptr;
  _PyCode_GetExtra((PyObject*)code, extra_index, (void**)&extra);
  return extra;
}

void destroy_extra_state(void* obj) {
  ExtraState* extra = (ExtraState*)obj;
  delete extra;
}

void set_extra_state(PyCodeObject* code, ExtraState* extra_state) {
  ExtraState* old_extra_state = get_extra_state(code);
  CHECK(extra_state == nullptr || old_extra_state != extra_state);
  _PyCode_SetExtra((PyObject*)code, extra_index, extra_state);
}

ExtraState* init_and_set_extra_state(PyCodeObject* code) {
  // Invariant - Extra state should not have been set before, therefore it
  // should be nullptr.
  CHECK(get_extra_state(code) == nullptr);
  ExtraState* extra_state = new ExtraState(code);
  NULL_CHECK(extra_state);
  set_extra_state(code, extra_state);
  // freed by destroy_extra_state (since we need to pass these objects to C)
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return extra_state;
}

static bool backend_match(PyObject* saved_backend, PyObject* backend) {
  // Pointer equality check for common case
  if (saved_backend != backend) {
    int result = PyObject_RichCompareBool(saved_backend, backend, Py_EQ);
    // Check for exception
    if (result == -1) {
      PyErr_Clear();
      return false;
    }
    return (result == 1);
  }
  return true;
}

void lookup(
    ExtraState* extra_state,
    FrameLocalsMapping* f_locals,
    PyObject* backend,
    PyObject** maybe_cached_code,
    const char** trace_annotation,
    bool is_skip_guard_eval_unsafe) {
  size_t index = 0;
  CacheEntry* found = nullptr;

  for (const auto& entry : extra_state->precompile_entries) {
    if (torch::dynamo::run_root_guard_manager(entry.root_mgr, f_locals)) {
      *maybe_cached_code = entry.code.ptr();
      return;
    }
  }

  for (CacheEntry& cache_entry : extra_state->cache_entry_list) {
    // Check backend. Py_False means run only mode.

    bool valid = Py_IsFalse(backend) ||
        backend_match(cache_entry.backend.ptr(), backend);

    if (valid) {
      try {
        if (is_skip_guard_eval_unsafe) {
          valid = torch::dynamo::run_root_guard_manager(
              cache_entry.diff_guard_root_mgr, f_locals);
        } else {
          valid = torch::dynamo::run_root_guard_manager(
              cache_entry.root_mgr, f_locals);
        }
      } catch (py::error_already_set& e) {
        if (guard_error_hook) {
          py::handle guard_error_hook_handle(guard_error_hook);
          py::handle f_locals_dict = (PyObject*)f_locals->to_dict();
          guard_error_hook_handle(
              cache_entry.guard_manager,
              cache_entry.code,
              f_locals_dict,
              index,
              index == extra_state->cache_entry_list.size() - 1);
        }
        // this function is called from C, so we cannot repropagate
        // the exception
        e.restore();
        *maybe_cached_code = nullptr;
        return;
      }
    }
    if (valid) {
      found = &cache_entry;
      break;
    }
    ++index;
  }
  if (found) {
    if (use_lru) {
      extra_state->move_to_front(found);
    }
    *maybe_cached_code = found->code.ptr();
    *trace_annotation = found->trace_annotation.c_str();
    return;
  }
  *maybe_cached_code = py::none().ptr();
}

CacheEntry* create_cache_entry(
    ExtraState* extra_state,
    PyObject* guarded_code,
    PyObject* backend) {
  std::list<CacheEntry>::iterator new_iter;
  if (use_lru) {
    extra_state->cache_entry_list.emplace_front(guarded_code, backend);
    new_iter = extra_state->cache_entry_list.begin();
  } else {
    extra_state->cache_entry_list.emplace_back(guarded_code, backend);
    new_iter = std::prev(extra_state->cache_entry_list.end());
  }
  new_iter->_owner = extra_state;
  new_iter->_owner_loc = new_iter;
  // Set guard_manager references to extra_state and CacheEntry
  // Warning: lifetime is controlled by C++!
  py::handle guard_manager = py::handle(guarded_code).attr("guard_manager");
  guard_manager.attr("cache_entry") =
      py::cast(*new_iter, py::return_value_policy::reference);
  guard_manager.attr("extra_state") =
      py::cast(extra_state, py::return_value_policy::reference);
  return &*new_iter;
}

py::list _debug_get_cache_entry_list(const py::handle& code_obj) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  py::list result;
  if (extra != nullptr) {
    for (CacheEntry& e : extra->cache_entry_list) {
      result.append(py::cast(e, py::return_value_policy::reference));
    }
  }
  return result;
}

py::object _debug_call_cache_entry_stable_callable(
    CacheEntry& cache_entry,
    py::dict f_locals,
    py::tuple arg_names,
    py::tuple args,
    py::object kwargs,
    bool use_diff_guard) {
  TORCH_CHECK(
      !cache_entry.code.is_none(),
      "cache entry stable callable is unavailable: cache entry is invalidated");
  TORCH_CHECK(
      PyCallable_Check(cache_entry.stable_callable.ptr()),
      "cache entry stable callable is unavailable: stable_callable is not callable");
  if (!kwargs.is_none()) {
    TORCH_CHECK_TYPE(
        PyDict_Check(kwargs.ptr()), "expected kwargs to be a dict or None");
    TORCH_CHECK(
        PyDict_Size(kwargs.ptr()) == 0,
        "cache entry stable callable only supports empty kwargs");
  }
  TORCH_CHECK(
      arg_names.size() == args.size(),
      "cache entry stable callable expected arg_names and args to have the same length");
  PyCodeObject* stable_code =
      reinterpret_cast<PyCodeObject*>(cache_entry.code.ptr());
  py::object stable_varnames =
      py::reinterpret_steal<py::object>(PyCode_GetVarnames(stable_code));
  TORCH_CHECK_TYPE(
      PyTuple_Check(stable_varnames.ptr()),
      "cache entry stable callable expected code varnames to be a tuple");
  TORCH_CHECK(
      static_cast<Py_ssize_t>(arg_names.size()) <=
          PyTuple_GET_SIZE(stable_varnames.ptr()),
      "cache entry stable callable arg_names exceed code varnames");
  for (size_t i = 0; i < arg_names.size(); ++i) {
    py::handle name = arg_names[i];
    TORCH_CHECK_TYPE(
        PyUnicode_Check(name.ptr()),
        "cache entry stable callable expected arg_names to contain strings");
    PyObject* expected_name = PyTuple_GET_ITEM(stable_varnames.ptr(), i);
    TORCH_CHECK(
        PyObject_RichCompareBool(name.ptr(), expected_name, Py_EQ) == 1,
        "cache entry stable callable arg_names must match code positional argument order");
    PyObject* local_value = PyDict_GetItemWithError(f_locals.ptr(), name.ptr());
    if (local_value == nullptr) {
      if (PyErr_Occurred()) {
        throw py::error_already_set();
      }
      TORCH_CHECK(
          false,
          "cache entry stable callable local/arg mismatch: missing guarded local");
    }
    TORCH_CHECK(
        local_value == args[i].ptr(),
        "cache entry stable callable local/arg mismatch");
  }

  void* root =
      use_diff_guard ? cache_entry.diff_guard_root_mgr : cache_entry.root_mgr;
  TORCH_CHECK(
      root != nullptr,
      "cache entry stable callable is unavailable: guard root is invalidated");
  if (!torch::dynamo::run_root_guard_manager_on_object(root, f_locals.ptr())) {
    TORCH_CHECK(false, "cache entry stable callable guard check failed");
  }

  PyObject* result =
      PyObject_CallObject(cache_entry.stable_callable.ptr(), args.ptr());
  if (result == nullptr) {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::object>(result);
}

py::object _debug_call_cache_entry_stable_callable_from_args(
    CacheEntry& cache_entry,
    py::tuple arg_names,
    py::tuple args,
    py::object kwargs,
    bool use_diff_guard) {
  TORCH_CHECK(
      arg_names.size() == args.size(),
      "cache entry stable callable expected arg_names and args to have the same length");
  py::dict f_locals;
  for (size_t i = 0; i < arg_names.size(); ++i) {
    py::handle name = arg_names[i];
    TORCH_CHECK_TYPE(
        PyUnicode_Check(name.ptr()),
        "cache entry stable callable expected arg_names to contain strings");
    f_locals[name] = args[i];
  }
  return _debug_call_cache_entry_stable_callable(
      cache_entry,
      f_locals,
      arg_names,
      args,
      std::move(kwargs),
      use_diff_guard);
}

PrecompileEntry::PrecompileEntry(py::object gm, py::object c)
    : guard_manager(std::move(gm)), code(std::move(c)) {
  TORCH_CHECK(
      PyCode_Check(code.ptr()), "Expecting CodeType from PrecompileEntry.");
  root_mgr =
      torch::dynamo::convert_to_root_guard_manager(guard_manager.attr("root"));
}

void _reset_precompile_entries(const py::handle& code_obj) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  py::list result;
  if (extra != nullptr) {
    extra->precompile_entries.clear();
  }
}

void _load_precompile_entry(
    const py::handle& code_obj,
    py::object guard_manager,
    py::object dynamo_code) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  py::list result;
  if (extra == nullptr) {
    extra = init_and_set_extra_state(code);
  }
  auto entry =
      PrecompileEntry(std::move(guard_manager), std::move(dynamo_code));
  extra->precompile_entries.push_back(std::move(entry));
}

void _set_lru_cache(py::object boolean) {
  if (py::cast<bool>(boolean)) {
    use_lru = true;
  } else {
    use_lru = false;
  }
}

py::list _debug_get_precompile_entries(const py::handle& code_obj) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  py::list result;
  if (extra != nullptr) {
    for (PrecompileEntry& e : extra->precompile_entries) {
      result.append(py::cast(e, py::return_value_policy::reference));
    }
  }
  return result;
}
