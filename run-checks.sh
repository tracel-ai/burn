#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# This script runs all `burn` checks locally
#
# Run `run-checks` using this command:
#
# ./scripts/run-checks environment
#
# where `environment` can assume **ONLY** the following values:
#
# - `std` to perform checks using `libstd`
# - `no_std` to perform checks on an embedded environment using `libcore`
# - `typos` to check for typos in the codebase
#
# If no `environment` value has been passed, run all checks.

# Compile run-checks binary
rustc scripts/run-checks.rs --crate-type bin --out-dir scripts

# Run binary passing the first input parameter, who is mandatory.
# If the input parameter is missing or wrong, it will be the `run-checks`
# binary which will be responsible of arising an error.
./scripts/run-checks $1
