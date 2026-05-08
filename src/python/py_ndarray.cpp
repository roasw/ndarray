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
        .def_property_readonly("shape", &ndarray::ndarray<float>::GetShape)
        .def_property_readonly("strides", &ndarray::ndarray<float>::GetStrides)
        .def_property_readonly("data_ptr",
                               [](const ndarray::ndarray<float> &self) {
                                   return reinterpret_cast<uintptr_t>(
                                       self.GetData());
                               })
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

    cls.def("__torch_function__", &TorchFunction, py::arg("func"),
            py::arg("types"), py::arg("args"), py::arg("kwargs") = py::none());

    module.def("from_torch", &FromTorchTensor, py::arg("tensor"));
    module.def("to_torch", &ToTorchTensor, py::arg("array"));
}
