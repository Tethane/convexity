CMAKE ?= cmake
CTEST ?= ctest
PYTHON ?= .venv/bin/python
BUILD_ROOT ?= build
BUILD_DIR ?= $(BUILD_ROOT)/debug
BUILD_TYPE ?= Debug
GENERATOR ?= $(shell if command -v ninja >/dev/null 2>&1; then echo Ninja; else echo 'Unix Makefiles'; fi)
PREFIX ?= $(CURDIR)/install

.PHONY: all configure build debug release test run install venv python python-run python-test clean help

all: debug

debug: BUILD_DIR := $(BUILD_ROOT)/debug
debug: BUILD_TYPE := Debug
debug: build

release: BUILD_DIR := $(BUILD_ROOT)/release
release: BUILD_TYPE := Release
release: build

build: configure
	$(CMAKE) --build "$(BUILD_DIR)"

configure:
	$(CMAKE) -S . -B "$(BUILD_DIR)" -G "$(GENERATOR)" \
		-DCMAKE_BUILD_TYPE="$(BUILD_TYPE)" \
		-DCONVEXITY_BUILD_TESTS=ON \
		-DCONVEXITY_BUILD_CLIENT=ON

test: debug
	$(CTEST) --test-dir "$(BUILD_DIR)" --output-on-failure

run: BUILD_DIR := $(BUILD_ROOT)/debug
run: BUILD_TYPE := Debug
run:
	@$(CMAKE) -S . -B "$(BUILD_DIR)" -G "$(GENERATOR)" \
		-DCMAKE_BUILD_TYPE="$(BUILD_TYPE)" \
		-DCONVEXITY_BUILD_TESTS=ON \
		-DCONVEXITY_BUILD_CLIENT=ON >/dev/null
	@$(CMAKE) --build "$(BUILD_DIR)" --target convexity_cpp_client >/dev/null
	@./$(BUILD_DIR)/convexity_cpp_client

install: release
	$(CMAKE) --install "$(BUILD_DIR)" --prefix "$(PREFIX)"

venv:
	python3 -m venv .venv

python: venv
	$(PYTHON) -m pip install --editable './python[dev]'

python-run: python
	$(PYTHON) python/examples/client.py

python-test: python
	$(PYTHON) -m pytest tests/python

clean:
	@for dir in "$(BUILD_DIR)" "$(BUILD_ROOT)" "$(BUILD_ROOT)/debug" "$(BUILD_ROOT)/release"; do \
		if [ -f "$$dir/CMakeCache.txt" ]; then \
			$(CMAKE) --build "$$dir" --target clean; \
		fi; \
	done

help:
	@echo "make          Build the default Debug configuration"
	@echo "make debug    Configure and build Debug"
	@echo "make release  Configure and build Release"
	@echo "make test     Build Debug and run CTest"
	@echo "make run      Build Debug and run the C++ client"
	@echo "make install  Build Release and install to ./install"
	@echo "make python   Create .venv and install the Python package"
	@echo "make python-test  Run Python tests"
	@echo "make clean    Remove native build artifacts"
