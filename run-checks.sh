#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# This script runs all `burn` checks locally. It may take around 15 minutes
# on the first run.
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
# - `examples` to check the examples compile
#
# If no `environment` value has been passed, run all checks except examples.

# Run binary passing the first input parameter, who is mandatory.
# If the input parameter is missing or wrong, it will be the `run-checks`
# binary which will be responsible of arising an error.
cargo xtask run-checks $1
