#include <convexity/core/version.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, module) {
    module.doc() = "Native Python bindings for Convexity core.";
    module.def("version", &convexity::version, "Return the Convexity library version.");
}
