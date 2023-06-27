#!/bin/bash

# This script is run before a PR is created.
# It is used to check that the code compiles and passes all tests.
# It is also used to check that the code is formatted correctly and passes clippy.

# Usage: ./run-checks.sh {all|no_std|std} (default: all)

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

# Function to handle errors
error_handler() {
    local exit_status=$?
    local line_number=$1
    local command=$2

    echo "Error on line $line_number"
    echo "Command '$command' exited with status $exit_status"
}

# Signal trap to call error_handler when a command fails
trap 'error_handler $LINENO $BASH_COMMAND' ERR

# Function to build and test no_std
build_and_test_no_std() {
    local dir=$1

    echo "$dir"
    cd $dir || exit

    echo "Build without defaults"
    cargo build --no-default-features

    echo "Test without defaults"
    cargo test --no-default-features

    echo "Build for WebAssembly"
    cargo build --no-default-features --target wasm32-unknown-unknown

    echo "Build for ARM"
    cargo build --no-default-features --target thumbv7m-none-eabi

    cd .. || exit
}

# Function to build and test all features
build_and_test_all_features() {
    local dir=$1

    echo "$dir"
    cd $dir || exit

    echo "Build with all defaults"
    cargo build --all-features

    echo "Test with all features"
    cargo test --all-features

    echo "Check documentation with all features"
    cargo doc --all-features

    cd .. || exit
}

# Set RUSTDOCFLAGS to treat warnings as errors for the documentation build
export RUSTDOCFLAGS="-D warnings"

# Run the checks for std and all features with std
std_func() {
    echo "Running std checks"

    cargo build --workspace
    cargo test --workspace
    cargo fmt --check --all
    cargo clippy -- -D warnings
    cargo doc --workspace

    # all features
    echo "Running all-features checks"
    build_and_test_all_features "burn-dataset"
}

# Run the checks for no_std
no_std_func() {
    echo "Running no_std checks"

    # Add wasm32 target for compiler.
    rustup target add wasm32-unknown-unknown
    rustup target add thumbv7m-none-eabi

    build_and_test_no_std "burn"
    build_and_test_no_std "burn-core"
    build_and_test_no_std "burn-common"
    build_and_test_no_std "burn-tensor"
    build_and_test_no_std "burn-ndarray"
    build_and_test_no_std "burn-no-std-tests"
}

# Save the script start time
start_time=$(date +%s)

# If no arguments were supplied or if it's empty, set the default as 'all'
if [ -z "${1-}" ]; then
    arg="all"
else
    arg=$1
fi

# Check the argument and call the appropriate functions
case $arg in
all)
    no_std_func
    std_func
    ;;
no_std)
    no_std_func
    ;;
std)
    std_func
    ;;
*)
    echo "Error: Invalid argument"
    echo "Usage: $0 {all|no_std|std}"
    exit 1
    ;;
esac

# Calculate and print the script execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Script executed in $execution_time seconds."

exit 0
