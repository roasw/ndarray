#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>

#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "container/ndarray.hpp"

namespace py = pybind11;

namespace {

template <typename T> inline constexpr bool kAlwaysFalse = false;

template <typename T> constexpr std::string_view TorchDTypeName() {
    if constexpr (std::is_same_v<T, float>) {
        return "float32";
    } else if constexpr (std::is_same_v<T, double>) {
        return "float64";
    } else {
        static_assert(kAlwaysFalse<T>, "Unsupported ndarray dtype");
    }
}

py::tuple IntTuple(const std::vector<int64_t> &values) {
    py::tuple out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = values[i];
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

int64_t ValueAtDim(const std::vector<int64_t> &values, int64_t dim) {
    const int64_t index =
        NormalizeDim(dim, static_cast<int64_t>(values.size()));
    return values[index];
}

std::vector<int64_t> ShapeFromIterable(py::iterable shape_like) {
    std::vector<int64_t> shape;
    for (py::handle value : shape_like) {
        shape.push_back(value.cast<int64_t>());
    }
    return shape;
}

template <typename T> py::tuple ShapeTuple(const ndarray::ndarray<T> &array) {
    return IntTuple(array.GetShape());
}

template <typename T> py::tuple StrideTuple(const ndarray::ndarray<T> &array) {
    return IntTuple(array.GetStrides());
}

template <typename T>
int64_t SizeAtDim(const ndarray::ndarray<T> &array, int64_t dim) {
    return ValueAtDim(array.GetShape(), dim);
}

template <typename T>
int64_t StrideAtDim(const ndarray::ndarray<T> &array, int64_t dim) {
    return ValueAtDim(array.GetStrides(), dim);
}

template <typename T> int64_t Numel(const ndarray::ndarray<T> &array) {
    const std::vector<int64_t> shape = array.GetShape();
    int64_t total = 1;
    for (const int64_t value : shape) {
        total *= value;
    }
    return total;
}

py::object TorchModule() { return py::module::import("torch"); }

py::object TorchDevice(c10::DeviceType device_type) {
    switch (device_type) {
    case c10::DeviceType::CPU:
        return TorchModule().attr("device")("cpu");
    case c10::DeviceType::CUDA:
        return TorchModule().attr("device")("cuda:0");
    default:
        throw std::runtime_error("Unsupported device");
    }
}

py::tuple DlpackDeviceTuple(c10::DeviceType device_type) {
    switch (device_type) {
    case c10::DeviceType::CPU:
        return py::make_tuple(1, 0);
    case c10::DeviceType::CUDA:
        return py::make_tuple(2, 0);
    default:
        throw std::runtime_error("Unsupported device for DLPack protocol");
    }
}

template <typename T> py::object TorchSize(const ndarray::ndarray<T> &array) {
    return TorchModule().attr("Size")(ShapeTuple(array));
}

template <typename T> py::object Dtype(const ndarray::ndarray<T> &) {
    return TorchModule().attr(TorchDTypeName<T>().data());
}

template <typename T>
at::Tensor ToTorchTensor(const ndarray::ndarray<T> &array);

template <typename T>
bool TryCastNdarrayToTorch(const py::handle &value, py::object &out) {
    try {
        auto array = py::cast<ndarray::ndarray<T>>(value);
        out = py::cast(ToTorchTensor(array));
        return true;
    } catch (const py::cast_error &) {
        return false;
    }
}

template <typename T> py::object Device(const ndarray::ndarray<T> &array) {
    return TorchDevice(array.GetDevice());
}

template <typename T>
bool IsContiguous(const ndarray::ndarray<T> &array, py::object memory_format) {
    at::Tensor tensor = ToTorchTensor(array);
    py::object torch_tensor = py::cast(tensor);
    if (memory_format.is_none()) {
        return torch_tensor.attr("is_contiguous")().cast<bool>();
    }
    py::dict kwargs;
    kwargs["memory_format"] = memory_format;
    return torch_tensor.attr("is_contiguous")(**kwargs).cast<bool>();
}

template <typename T> py::tuple DlpackDevice(const ndarray::ndarray<T> &array) {
    return DlpackDeviceTuple(array.GetDevice());
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

template <typename T>
py::capsule ToDLPackCapsule(const ndarray::ndarray<T> &array) {
    DLManagedTensor *managed = array.ToDLPack();
    if (!managed) {
        throw std::runtime_error("Cannot export empty ndarray to DLPack");
    }

    return py::capsule(managed, "dltensor", DlpackCapsuleDestructor);
}

template <typename T>
ndarray::ndarray<T> FromTorchTensor(const at::Tensor &tensor) {
    return ndarray::ndarray<T>::FromDLPack(at::toDLPack(tensor));
}

py::object FromTorchTensorDynamic(const at::Tensor &tensor) {
    switch (tensor.scalar_type()) {
    case at::kFloat:
        return py::cast(FromTorchTensor<float>(tensor));
    case at::kDouble:
        return py::cast(FromTorchTensor<double>(tensor));
    default:
        throw py::type_error(
            "from_torch expects torch.float32 or torch.float64");
    }
}

template <typename T>
at::Tensor ToTorchTensor(const ndarray::ndarray<T> &array) {
    DLManagedTensor *managed = array.ToDLPack();
    if (managed == nullptr) {
        return at::Tensor();
    }
    return at::fromDLPack(managed);
}

py::object ToTorchDynamic(const py::handle &value) {
    py::object out;
    if (TryCastNdarrayToTorch<float>(value, out) ||
        TryCastNdarrayToTorch<double>(value, out)) {
        return out;
    }

    throw py::type_error("to_torch expects ndarray_f32 or ndarray_f64");
}

py::object MaybeToTorch(const py::handle &value) {
    try {
        return ToTorchDynamic(value);
    } catch (const py::type_error &) {
        return py::reinterpret_borrow<py::object>(value);
    }
}

template <typename T>
py::class_<ndarray::ndarray<T>> BindNdarrayClass(py::module_ &module,
                                                 std::string_view name) {
    py::class_<ndarray::ndarray<T>> cls(module, name.data());
    cls.def(py::init<>())
        .def(py::init([](py::iterable shape_like) {
                 return ndarray::ndarray<T>(ShapeFromIterable(shape_like),
                                            c10::DeviceType::CPU);
             }),
             py::arg("shape"))
        .def_static("from_torch", &FromTorchTensor<T>, py::arg("tensor"))
        .def("to_torch", &ToTorchTensor<T>)
        .def(
            "__dlpack__",
            [](const ndarray::ndarray<T> &self, py::object) {
                return ToDLPackCapsule(self);
            },
            py::arg("stream") = py::none())
        .def("__dlpack_device__", &DlpackDevice<T>)
        .def_property_readonly("ndim", &ndarray::ndarray<T>::GetNdim)
        .def("dim", &ndarray::ndarray<T>::GetNdim)
        .def("size", &TorchSize<T>)
        .def("size", &SizeAtDim<T>, py::arg("dim"))
        .def("stride",
             [](const ndarray::ndarray<T> &self) { return StrideTuple(self); })
        .def("stride", &StrideAtDim<T>, py::arg("dim"))
        .def("numel", &Numel<T>)
        .def("is_contiguous", &IsContiguous<T>,
             py::arg("memory_format") = py::none())
        .def("data_ptr",
             [](const ndarray::ndarray<T> &self) {
                 return reinterpret_cast<uintptr_t>(self.GetData());
             })
        .def_property_readonly("shape", &TorchSize<T>)
        .def_property_readonly("strides", &StrideTuple<T>)
        .def_property_readonly("dtype", &Dtype<T>)
        .def_property_readonly("device", &Device<T>)
        .def("clone", &ndarray::ndarray<T>::Clone)
        .def("transpose", &ndarray::ndarray<T>::Transpose)
        .def("__add__", &ndarray::ndarray<T>::operator+)
        .def("__sub__", &ndarray::ndarray<T>::operator-)
        .def("__mul__", &ndarray::ndarray<T>::operator*)
        .def("__truediv__", &ndarray::ndarray<T>::operator/)
        .def("__repr__", [](const ndarray::ndarray<T> &self) {
            return "ndarray(shape=" +
                   py::str(py::cast(self.GetShape())).cast<std::string>() +
                   ", dtype=" + std::string(TorchDTypeName<T>()) +
                   ", device=cpu)";
        });
    return cls;
}

py::object TorchFunction(py::object, py::object func, py::object,
                         py::tuple args, py::object kwargs_obj);

template <typename T>
void BindTorchFunction(py::class_<ndarray::ndarray<T>> &cls) {
    py::object classmethod_obj =
        py::module::import("builtins").attr("classmethod");
    py::object torch_function =
        py::cpp_function(&TorchFunction, py::name("__torch_function__"),
                         py::arg("cls"), py::arg("func"), py::arg("types"),
                         py::arg("args"), py::arg("kwargs") = py::none());
    cls.attr("__torch_function__") = classmethod_obj(torch_function);
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
    auto cls_f32 = BindNdarrayClass<float>(module, "ndarray_f32");
    auto cls_f64 = BindNdarrayClass<double>(module, "ndarray_f64");
    BindTorchFunction(cls_f32);
    BindTorchFunction(cls_f64);

    module.def("from_torch", &FromTorchTensorDynamic, py::arg("tensor"));
    module.def("to_torch", &ToTorchDynamic, py::arg("array"));
}
