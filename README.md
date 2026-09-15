# Convexity

Convexity is a minimal quantitative research and trading SDK foundation. The current release is intentionally small: a C++ shared library exposes one version function, a C++ client links to it, and a Python extension calls the same native implementation.

The dependency direction is:

```text
Python binding -> Convexity C++ core <- future adapters
```

No market data, trading, backtesting, networking, venue adapters, or speculative abstractions are included yet.

## Prerequisites

The native build needs a C++ compiler, CMake 3.24 or newer, Ninja, and a CMake-configured GoogleTest installation. On Ubuntu, the prerequisites can be installed with:

```bash
sudo apt install build-essential cmake ninja-build libgtest-dev python3-venv
```

Python development uses Python 3.10 or newer and a virtual environment. The Python build obtains `scikit-build-core` and `pybind11` from `python/pyproject.toml`.

## C++ development

Configure a Debug build with Ninja:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCONVEXITY_BUILD_TESTS=ON \
  -DCONVEXITY_BUILD_CLIENT=ON
```

Compile the shared library, C++ client, and tests:

```bash
cmake --build build
```

Run the C++ client:

```bash
./build/convexity_cpp_client
```

Run the C++ tests through CTest:

```bash
ctest --test-dir build --output-on-failure
```

The same workflow is available through the Makefile:

```bash
make              # Debug build
make test
make run
make release
make clean
make help
```

To enable AddressSanitizer and UndefinedBehaviorSanitizer for a development build, add `-DCONVEXITY_ENABLE_SANITIZERS=ON` to the configure command.

The install tree exports the target as `convexity::core`, ready for a consumer using `find_package(convexity CONFIG REQUIRED)`:

```bash
cmake --install build --prefix "$PWD/install"
```

## Python development

Create and activate an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --editable "./python[dev]"
```

The editable install builds the native Python extension from the same C++ core. Run the Python client:

```bash
python python/examples/client.py
```

Run Python tests:

```bash
python -m pytest tests/python
```

The standalone CMake build does not build Python by default. To build the binding directly with an existing pybind11 installation, configure with `-DCONVEXITY_BUILD_PYTHON=ON` and provide `pybind11_DIR` when needed, for example `$(python -m pybind11 --cmakedir)`.

## Layout

```text
include/convexity/   Public C++ headers
src/core/             Private C++ implementation
apps/cpp_client/      External-style C++ consumer
bindings/python/      Thin pybind11 module
python/               Python package, client, and tests
tests/cpp/            GoogleTest tests
cmake/                Small project CMake helpers and package config
```

This repository is a local Git repository. Future capabilities can be added as independent components without making the core depend on venue-specific adapters.
