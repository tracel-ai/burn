CI notes: cargo-audit advisory handling
=====================================

This file documents how CI handles RustSec advisories detected by `cargo audit` and the remediation procedure.

Current situation
-----------------

CI runs `cargo audit` in `.github/workflows/dependencies.yml`. If advisories are found the job will fail and produce `target/audit/rustsec-report.json` and `target/audit/rustsec-report.txt` which are uploaded as artifacts for triage.

Known advisory
--------------

- RUSTSEC-2024-0375 — `atty` is unmaintained, pulled in transitively via `criterion` used by `gaba-native-kernels`.

Remediation steps
-----------------

1. Try updating the transitive dependency:

   cargo update -p criterion
   cargo update -p atty

2. Re-run cargo audit and verify the advisory is gone.

3. If update is not possible, follow these options in priority order:
   - Replace the crate using `atty` or `criterion` in the affected crate with a maintained alternative.
   - Vendor a patched version of the crate.
   - Document the accepted risk and schedule a re-check in 30 days.

Opening an issue
----------------

Use the built-in issue template `/.github/ISSUE_TEMPLATE/rustsec-remediation.md` when opening an issue to track remediation. A suggested remediation file already exists at `.github/ISSUES/RUSTSEC-2024-0375-TRACK.md`.

Re-check cadence
----------------

Re-run `cargo audit` weekly until the advisory is resolved or an exception is documented.
