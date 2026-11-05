# Local xtask v6 development

This migration branch uses the local framework at
`/Users/sylvain/src/github/tracel/xtask/crates/tracel-xtask` (version 6.0.0).
The xtask package's `bazel` feature enables native Bazel support and is selected
by this branch's default features. `cargo check -p xtask --no-default-features`
checks the Cargo-only configuration; place its target directory on Tracel.

The standalone xtask CLI accepts `--cargo`, `--bazel`, and `--auto` before the
subcommand, including selection of the frontend compiler. A Cargo alias has
already compiled the frontend with Cargo, so it cannot measure both scopes.

Burn's custom grouped test commands dispatch through xtask's backend API. Use
`test --ci selected --target crates --only burn-nn --features burn-core/flex unit`
for focused crate tests. The normal CI shards remain available.

The official Rust benchmark lives in the xtask repository, not this checkout:
`/Users/sylvain/src/github/tracel/xtask/docs/build-benchmarks.md`.
It covers workspace builds, CPU/CI tests, retained-cache editing sessions, and
CubeCL changes validated through Burn's Metal shard. Invoke its Bash launcher
from that repository and use Tracel for `--data-root` and `--output-root`.

For linked development pass both `--workspace` (this repository) and
`--cubecl-workspace /Volumes/Tracel/src/github/tracel/cubecl`. The runner patches
all CubeCL packages in private source snapshots, verifies Cargo resolution, and
leaves this checkout and its pinned production dependencies unchanged.
