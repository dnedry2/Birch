#ifndef __B_PYTHON_HXX_
#define __B_PYTHON_HXX_

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <string>
#include <vector>

namespace Python {
	struct PyObj;
	template<class T> T MakeVar(PyObj* obj);

	struct PyObj {
		PyObject* Obj;
		bool Free;

		PyObj() : Obj(nullptr), Free(false) { }
		PyObj(PyObject* obj, bool free = true) : Obj(obj), Free(free) { }
		~PyObj() {
			if (Free) {
				Py_DECREF(Obj);
			}
		}

		template<class T>
		operator T() {
  			return MakeVar<T>(this);
		}

		PyObj(const PyObj& other) {
			Obj  = other.Obj;
			Free = other.Free;
		}
	};

	template<class T> PyObj MakeObj(T var);
	template<> PyObj MakeObj<PyObj>(PyObj var) {
		return var;
	}
	template<> PyObj MakeObj<int>(int var) {
		return PyObj(PyLong_FromLong(var));
	}
	template<> PyObj MakeObj<PyObject*>(PyObject* var) {
		return PyObj(var);
	}

	template<class T> PyObj MakeArrayObj(T* array, uint len);
	template<> PyObj MakeArrayObj<double>(double* array, uint len) {
		npy_intp dims = len;
		return PyObj(PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, reinterpret_cast<void*>(array)), false);
	}
	template<> PyObj MakeArrayObj<char*>(char** array, uint len) {
		npy_intp dims = len;
		return PyObj(PyArray_SimpleNewFromData(1, &dims, NPY_STRING, reinterpret_cast<void*>(array)), false);
	}

	template<> int MakeVar<int>(PyObj* obj) {
		return PyLong_AsLong(obj->Obj);
	}
	template<> std::string MakeVar<std::string>(PyObj* obj) {
		return std::string(PyUnicode_AsUTF8(obj->Obj));
	}

	struct PyDict {
		//void Add(PyObj* key, PyObj* value) {
		//	PyDict_SetItem(dict, key->Obj, value->Obj);
		//}
		void Add(const char* key, PyObj value) {
			PyDict_SetItemString(dict, key, value.Obj);
		}

		PyObject* Get() {
			return dict;
		}

		operator PyObj() {
			return PyObj(dict, false);
		}

		PyDict() {
			dict = PyDict_New();
		}
		~PyDict() {
			Py_DECREF(dict);
		}
	private:
		PyObject* dict;
	};

	namespace internal {
		struct tuple_packer {
			template<class T, class... ARGS>
			void pack(PyTupleObject* tuple, int index, T value, ARGS... args) {
				auto var = MakeObj(value);

				PyTuple_SetItem((PyObject*)tuple, idx, var.Obj);
				pack(tuple, ++idx, args...);
			}
			void pack(PyTupleObject* tuple, int index) { }

			int idx = 0;
		};

		template<class... ARGS>
		PyObj call(PyObj* module, const char* function, ARGS... args) {
			PyObject* func = PyObject_GetAttrString(module->Obj, function);

			auto cnt = sizeof...(args);

			PyTupleObject* argTuple = (PyTupleObject*)PyTuple_New(cnt);

			tuple_packer packer;
			packer.pack(argTuple, 0, args...);

			auto result = PyObj(PyObject_CallObject(func, (PyObject*)argTuple));

			Py_DECREF(argTuple);
			Py_DECREF(func);

			return result;
		}
	};

	template<class... ARGS>
	PyObj Call(PyObj* module, const char* function, ARGS... args) {
		return internal::call(module, function, args...);
	}
	PyObj Call(PyObj* module, const char* function) {
		PyObject* func = PyUnicode_FromString(function);

		PyObject* result = PyObject_CallMethodNoArgs(module->Obj, func);

		Py_DECREF(func);

		return result;
	}
	template<class T>
	T Call(PyObj* module, const char* function) {
		auto result = Call(module, function);
		return MakeVar<T>(&result);
	}

	void Init(const char* path = ".") {
		setenv("PYTHONPATH", path, 1);

		Py_Initialize();

		_import_array();
	}
	void Shutdown() {
		Py_Finalize();
	}

	PyObj LoadModule(const char* name) {
		PyObject* moduleString = PyUnicode_FromString(name);
		PyObject* module = PyImport_Import(moduleString);

		Py_DECREF(moduleString);

		return PyObj(module);
	}

	void Free(PyObj* object) {
		object->Free = true;

		delete object;
	}

	bool HasFunction(PyObj* module, const char* function) {
		PyObject* func = PyObject_GetAttrString(module->Obj, function);

		if (func && PyCallable_Check(func)) {
			Py_DECREF(func);
			return true;
		}

		if (PyErr_Occurred()) {
			PyErr_Clear();
		}

		return false;
	}

	bool HasFunctions(PyObj* module, const std::vector<std::string>& functions) {
		for (auto function : functions)
			if (!HasFunction(module, function.c_str()))
				return false;

		return true;
	}
};

#endif