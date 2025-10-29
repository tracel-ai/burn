Title: Track remediation for RUSTSEC-2024-0375 (atty unmaintained)

Status: open

Summary
-------

This file tracks remediation steps for advisory RUSTSEC-2024-0375 affecting `atty` (unmaintained) pulled in via `criterion` in `gaba-native-kernels`.

Work items
----------
- [ ] Attempt cargo update on `criterion` and `atty` (run `cargo update -p criterion` and `cargo update -p atty`).
- [ ] If update succeeds and `cargo audit` clears, commit updated Cargo.lock and close this file/issue.
- [ ] If update not possible, explore replacing `criterion` in `gaba-native-kernels` or patching with a fork.
- [ ] Document decision and schedule re-check in 30 days.

Local commands
--------------

```bash
# inspect current advisory
cargo install --locked cargo-audit || true
cargo audit --json > target/audit/rustsec-report.json || true
cat target/audit/rustsec-report.txt

# attempt updates
cargo update -p criterion
cargo update -p atty

# re-run audit
cargo audit
```

Notes
-----
CI currently reports the advisory. We must either upgrade, replace, or document/accept the risk with a scheduled re-check.
