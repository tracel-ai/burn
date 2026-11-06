#!/bin/bash
set -euo pipefail

checks_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$checks_dir/../../../../.." && pwd)"
checks_toolchain="${ADAMW_CHECKS_TOOLCHAIN:-beta}"
checks_cargo="$(rustup which --toolchain "$checks_toolchain" cargo)"
checks_name="${ADAMW_CHECKS_LOG_NAME:-decorators}"
checks_features="${ADAMW_CHECKS_FEATURES:-}"
checks_args=(test --locked --manifest-path "$checks_dir/Cargo.toml"
    --config "$repo_dir/target/adamw-foreach-local.toml" --test decorators)
if [[ -n "$checks_features" ]]; then
    checks_args+=(--features "$checks_features")
fi
checks_raw_log="$checks_dir/logs/$checks_name.raw.log"
checks_summary="$checks_dir/logs/$checks_name.log"
mkdir -p "$checks_dir/logs"
cd "$checks_dir"

export RUSTC="$(rustup which --toolchain "$checks_toolchain" rustc)"
export RUSTDOC="$(rustup which --toolchain "$checks_toolchain" rustdoc)"
export RUSTC_WRAPPER=
export RUSTC_WORKSPACE_WRAPPER=
export CARGO_TARGET_DIR="$repo_dir/target/adamw-foreach-decorators"
export BURN_DEVICE=cpu
unset CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER

{
    date -u '+UTC %Y-%m-%dT%H:%M:%SZ'
    uname -sm
    "$checks_cargo" --version
    "$RUSTC" --version
    "$RUSTDOC" --version
    printf 'toolchain=%s\ncargo=%s\nrustc=%s\nrustdoc=%s\n' "$checks_toolchain" "$checks_cargo" "$RUSTC" "$RUSTDOC"
    printf 'target=%s\noverride=%s\nwrappers=empty\nBURN_DEVICE=%s\n' "$CARGO_TARGET_DIR" "$repo_dir/target/adamw-foreach-local.toml" "$BURN_DEVICE"
    printf 'harness_lock_sha256=%s\n' "$(shasum -a 256 Cargo.lock | awk '{print $1}')"
    printf 'features=%s\n' "$checks_features"
} > "$checks_raw_log"

# Capture cargo's status immediately; tee's status must not turn a failed test into success.
set +e
"$checks_cargo" "${checks_args[@]}" -- --nocapture --test-threads=1 \
    2>&1 | tee -a "$checks_raw_log"
checks_pipeline_status=("${PIPESTATUS[@]}")
set -e
checks_test_status="${checks_pipeline_status[0]}"
checks_tee_status="${checks_pipeline_status[1]}"

# Keep the complete runtime output, tool versions and compile outcome in the review artifact.
# Full compiler output stays local, excluded by .gitignore, and is identified by its hash.
awk '
    /^running [0-9]+ tests?/ { runtime = 1 }
    runtime { print; next }
    /^UTC |^Darwin |^Linux |^(cargo|rustc|rustdoc) |^[a-zA-Z_]+=|^ *Finished |^ *Running |^error/ { print }
' "$checks_raw_log" > "$checks_summary"
{
    printf 'cargo_test_exit_status=%s\ntee_exit_status=%s\n' "$checks_test_status" "$checks_tee_status"
    printf 'raw_log_sha256=%s\n' "$(shasum -a 256 "$checks_raw_log" | awk '{print $1}')"
} >> "$checks_summary"
printf 'Result summary: %s\n' "$checks_summary"
if (( checks_test_status != 0 )); then
    exit "$checks_test_status"
fi
exit "$checks_tee_status"
