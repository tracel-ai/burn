#!/usr/bin/env pwsh

# Exit immediately if a command exits with a non-zero status.
$ErrorActionPreference = "Stop"

# This script runs all `burn` checks locally. It may take around 15 minutes
# on the first run.
#
# Run `run-checks` using this command:
#
# ./run-checks.ps1 environment
#
# where `environment` can assume **ONLY** the following values:
#
# - `std` to perform validation using `libstd`
  # - `no-std` to perform validation on an embedded environment using `libcore`
  # - `all` to perform both std and no-std validation
#
# If no `environment` value has been passed, default to `all`.
$exec_env = if ($args.Count -ge 1) { $args[0] } else { "all" }

# Run the cargo xtask command with the specified environment
cargo xtask --execution-environment $exec_env validate
