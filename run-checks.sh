#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# This script runs all `burn` checks locally. It may take around 15 minutes
# on the first run.
#
# Run `run-checks` using this command:
#
# ./run-checks.sh environment
#
# where `environment` can assume **ONLY** the following values:
#
# - `std` to perform validation using `libstd`
# - `no-std` to perform validation on an embedded environment using `libcore`
# - `all` to perform both std and no-std validation
#
# If no `environment` value has been passed.
exec_env=${1:-all}

cargo xtask --execution-environment "$exec_env" validate
