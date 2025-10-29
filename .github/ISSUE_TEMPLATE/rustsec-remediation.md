---
name: RustSec advisory remediation
about: Template to open an issue for remediating RustSec advisories found by CI (cargo-audit)
title: "Remediate RustSec advisory: RUSTSEC-2024-0375 (atty unmaintained)"
labels: security, dependencies
assignees: ""
---

Description
-----------

CI's dependency audit detected advisory RUSTSEC-2024-0375 (unmaintained `atty` crate) which is pulled in transitively via `criterion` used by `gaba-native-kernels`.

Steps to reproduce
------------------

Run locally from repo root:

```bash
cargo install --locked cargo-audit || true
cargo audit --json > target/audit/rustsec-report.json || true
jq . target/audit/rustsec-report.json
```

Proposed remediation plan
-------------------------
1. Try to update `criterion` (or `atty`) to a version that removes the unmaintained crate:

   ```bash
   # attempt to update criterion
   cargo update -p criterion
   # or update atty directly
   cargo update -p atty
   ```

2. If update not possible, consider replacing `criterion` in the affected crate with an alternative or vendor a patched version.

3. If the advisory is acceptable short-term, document the rationale and schedule a re-check in 30 days.

Acceptance criteria
-------------------
- CI `cargo audit` passes (no advisories), or a documented exception + scheduled re-evaluation exists.

Notes
-----
- Advisory: RUSTSEC-2024-0375
- URL: https://rustsec.org/advisories/RUSTSEC-2024-0375
