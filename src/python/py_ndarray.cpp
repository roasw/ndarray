#include <cstdint>
#include <stdexcept>
#include <string>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>

#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "container/ndarray.hpp"

namespace py = pybind11;

namespace {

py::tuple ShapeTuple(const ndarray::ndarray<float> &array) {
    const std::vector<int64_t> shape = array.GetShape();
    py::tuple out(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        out[i] = shape[i];
    }
    return out;
}

py::tuple StrideTuple(const ndarray::ndarray<float> &array) {
    const std::vector<int64_t> strides = array.GetStrides();
    py::tuple out(strides.size());
    for (size_t i = 0; i < strides.size(); ++i) {
        out[i] = strides[i];
    }
    return out;
}

int64_t NormalizeDim(int64_t dim, int64_t ndim) {
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw py::index_error("dimension out of range");
    }
    return dim;
}

int64_t SizeAtDim(const ndarray::ndarray<float> &array, int64_t dim) {
    const std::vector<int64_t> shape = array.GetShape();
    const int64_t index = NormalizeDim(dim, static_cast<int64_t>(shape.size()));
    return shape[index];
}

int64_t StrideAtDim(const ndarray::ndarray<float> &array, int64_t dim) {
    const std::vector<int64_t> strides = array.GetStrides();
    const int64_t index =
        NormalizeDim(dim, static_cast<int64_t>(strides.size()));
    return strides[index];
}

int64_t Numel(const ndarray::ndarray<float> &array) {
    const std::vector<int64_t> shape = array.GetShape();
    int64_t total = 1;
    for (const int64_t value : shape) {
        total *= value;
    }
    return total;
}

py::object TorchModule() { return py::module::import("torch"); }

py::object TorchSize(const ndarray::ndarray<float> &array) {
    return TorchModule().attr("Size")(ShapeTuple(array));
}

py::object Dtype(const ndarray::ndarray<float> &) {
    return TorchModule().attr("float32");
}

at::Tensor ToTorchTensor(const ndarray::ndarray<float> &array);

py::object Device(const ndarray::ndarray<float> &array) {
    switch (array.GetDevice()) {
    case c10::DeviceType::CPU:
        return TorchModule().attr("device")("cpu");
    case c10::DeviceType::CUDA:
        return TorchModule().attr("device")("cuda:0");
    default:
        throw std::runtime_error("Unsupported device");
    }
}

bool IsContiguous(const ndarray::ndarray<float> &array,
                  py::object memory_format) {
    at::Tensor tensor = ToTorchTensor(array);
    py::object torch_tensor = py::cast(tensor);
    if (memory_format.is_none()) {
        return torch_tensor.attr("is_contiguous")().cast<bool>();
    }
    py::dict kwargs;
    kwargs["memory_format"] = memory_format;
    return torch_tensor.attr("is_contiguous")(**kwargs).cast<bool>();
}

py::tuple DlpackDevice(const ndarray::ndarray<float> &array) {
    switch (array.GetDevice()) {
    case c10::DeviceType::CPU:
        return py::make_tuple(1, 0);
    case c10::DeviceType::CUDA:
        return py::make_tuple(2, 0);
    default:
        throw std::runtime_error("Unsupported device for DLPack protocol");
    }
}

void DlpackCapsuleDestructor(PyObject *capsule) {
    if (PyCapsule_IsValid(capsule, "used_dltensor") != 0) {
        return;
    }

    DLManagedTensor *tensor = static_cast<DLManagedTensor *>(
        PyCapsule_GetPointer(capsule, "dltensor"));
    if (tensor == nullptr) {
        PyErr_Clear();
        return;
    }

    if (tensor->deleter != nullptr) {
        tensor->deleter(tensor);
    }
}

py::capsule ToDLPackCapsule(const ndarray::ndarray<float> &array) {
    DLManagedTensor *managed = array.ToDLPack();
    if (!managed) {
        throw std::runtime_error("Cannot export empty ndarray to DLPack");
    }

    return py::capsule(managed, "dltensor", DlpackCapsuleDestructor);
}

ndarray::ndarray<float> FromTorchTensor(const at::Tensor &tensor) {
    return ndarray::ndarray<float>::FromDLPack(at::toDLPack(tensor));
}

at::Tensor ToTorchTensor(const ndarray::ndarray<float> &array) {
    DLManagedTensor *managed = array.ToDLPack();
    if (managed == nullptr) {
        return at::Tensor();
    }
    return at::fromDLPack(managed);
}

py::object MaybeToTorch(const py::handle &value) {
    try {
        ndarray::ndarray<float> array =
            py::cast<ndarray::ndarray<float>>(value);
        return py::cast(ToTorchTensor(array));
    } catch (const py::cast_error &) {
        return py::reinterpret_borrow<py::object>(value);
    }
}

py::object TorchFunction(py::object, py::object func, py::object,
                         py::tuple args, py::object kwargs_obj) {

    py::tuple converted_args(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        converted_args[i] = MaybeToTorch(args[i]);
    }

    py::dict kwargs;
    if (!kwargs_obj.is_none()) {
        py::dict kwargs_in = kwargs_obj.cast<py::dict>();
        for (auto item : kwargs_in) {
            kwargs[item.first] = MaybeToTorch(item.second);
        }
    }

    return func(*converted_args, **kwargs);
}

} // namespace

PYBIND11_MODULE(_ndarray, module) {
    py::class_<ndarray::ndarray<float>> cls(module, "ndarray");
    cls.def(py::init<>())
        .def(py::init([](py::iterable shape_like) {
                 std::vector<int64_t> shape;
                 for (py::handle value : shape_like) {
                     shape.push_back(value.cast<int64_t>());
                 }
                 return ndarray::ndarray<float>(shape, c10::DeviceType::CPU);
             }),
             py::arg("shape"))
        .def_static("from_torch", &FromTorchTensor, py::arg("tensor"))
        .def("to_torch", &ToTorchTensor)
        .def(
            "__dlpack__",
            [](const ndarray::ndarray<float> &self, py::object) {
                return ToDLPackCapsule(self);
            },
            py::arg("stream") = py::none())
        .def("__dlpack_device__", &DlpackDevice)
        .def_property_readonly("ndim", &ndarray::ndarray<float>::GetNdim)
        .def("dim", &ndarray::ndarray<float>::GetNdim)
        .def("size", &TorchSize)
        .def("size", &SizeAtDim, py::arg("dim"))
        .def("stride",
             [](const ndarray::ndarray<float> &self) {
                 return StrideTuple(self);
             })
        .def("stride", &StrideAtDim, py::arg("dim"))
        .def("numel", &Numel)
        .def("is_contiguous", &IsContiguous,
             py::arg("memory_format") = py::none())
        .def("data_ptr",
             [](const ndarray::ndarray<float> &self) {
                 return reinterpret_cast<uintptr_t>(self.GetData());
             })
        .def_property_readonly("shape", &TorchSize)
        .def_property_readonly("strides", &StrideTuple)
        .def_property_readonly("dtype", &Dtype)
        .def_property_readonly("device", &Device)
        .def("clone", &ndarray::ndarray<float>::Clone)
        .def("transpose", &ndarray::ndarray<float>::Transpose)
        .def("__add__", &ndarray::ndarray<float>::operator+)
        .def("__sub__", &ndarray::ndarray<float>::operator-)
        .def("__mul__", &ndarray::ndarray<float>::operator*)
        .def("__truediv__", &ndarray::ndarray<float>::operator/)
        .def("__repr__", [](const ndarray::ndarray<float> &self) {
            return "ndarray(shape=" +
                   py::str(py::cast(self.GetShape())).cast<std::string>() +
                   ", device=cpu)";
        });

    py::object classmethod_obj =
        py::module::import("builtins").attr("classmethod");
    py::object torch_function =
        py::cpp_function(&TorchFunction, py::name("__torch_function__"),
                         py::arg("cls"), py::arg("func"), py::arg("types"),
                         py::arg("args"), py::arg("kwargs") = py::none());
    cls.attr("__torch_function__") = classmethod_obj(torch_function);

    module.def("from_torch", &FromTorchTensor, py::arg("tensor"));
    module.def("to_torch", &ToTorchTensor, py::arg("array"));
}
