#!/bin/bash

# This script is run before a PR is created.
# It is used to check that the code compiles and passes all tests.
# It is also used to check that the code is formatted correctly and passes clippy.

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

    echo "Test with defaults"
    cargo test --all-features

    cd .. || exit
}

# Save the script start time
start_time=$(date +%s)

# Add wasm32 target for compiler.
rustup target add wasm32-unknown-unknown
rustup target add thumbv7m-none-eabi

# TODO decide if we should "cargo clean" here.
cargo build --workspace
cargo test --workspace
cargo fmt --check --all
cargo clippy -- -D warnings

# no_std tests
build_and_test_no_std "burn"
build_and_test_no_std "burn-core"
build_and_test_no_std "burn-common"
build_and_test_no_std "burn-tensor"
build_and_test_no_std "burn-ndarray"
build_and_test_no_std "burn-no-std-tests"

# all features tests
build_and_test_all_features "burn-dataset"

# Calculate and print the script execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Script executed in $execution_time seconds."
