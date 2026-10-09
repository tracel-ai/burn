# Burn

Burn is a Rust tensor library and deep learning framework for training and inference across multiple
backends. Changes must preserve the public API's semantics across supported devices, dtypes, and
execution modes.

## Start Here

- Read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution policy, change ownership, and review
  expectations.
- Use the [Contributor Book](contributor-book/src/SUMMARY.md) for architecture and development
  workflows, and the [Burn Book](burn-book/src/SUMMARY.md) for user-facing behavior and examples.
- Check API definitions, feature flags, and working examples in this checkout. APIs from an older
  release or another project may not match the code here.

## Repository Layout

- `crates/burn/`: the public entry point and feature selection for applications.
- `crates/burn-tensor/`, `burn-backend/`, and `burn-dispatch/`: the tensor API, backend contracts,
  and runtime dispatch.
- `crates/burn-core/`, `burn-nn/`, `burn-optim/`, `burn-train/`, and `burn-dataset/`: modules,
  neural network layers, optimization, training, and data handling.
- `crates/burn-cubecl/`: shared CubeCL backend operations and kernels. `burn-cuda/`, `burn-rocm/`,
  `burn-wgpu/`, and `burn-cpu/` select runtimes; `burn-flex/` is the pure-Rust CPU backend.
- `crates/burn-autodiff/`, `burn-fusion/`, `burn-cubecl-fusion/`, `burn-ir/`, and `burn-router/`:
  differentiation, fusion, operation recording, and routing.
- `crates/burn-store/` and `burn-pack/`: model weight storage, import, and the burnpack format.
- `crates/burn-backend-tests/`: shared tensor, autodiff, and fusion test suites.
- `examples/`: runnable applications and examples used by the books.
- `burn-book/` and `contributor-book/`: user and contributor documentation.
- `xtask/`: repository checks, tests, builds, and book tooling.

## Architecture and Changes

For tensor operations, trace the public API through the bridge and dispatch layers to the backend
implementation. Consider the applicable autodiff, IR, fusion, and router paths as well. The
[operation guide](contributor-book/src/guides/adding-a-new-operation-to-burn.md) explains where each
part belongs and which tests it needs.

- Keep changes focused. Separate mechanical moves and renames from behavior changes so the diff can
  be reviewed independently.
- Preserve feature isolation and `no_std` support where the affected crates provide it.
- For backend changes, account for supported layouts, dtypes, and hardware capabilities. State
  unsupported cases explicitly rather than assuming the development machine represents every device.
- Update documentation and examples in the same change when behavior or APIs change.

### Backends and Kernels

- `burn-cuda`, `burn-rocm`, `burn-wgpu`, and `burn-cpu` select runtimes for the shared `burn-cubecl`
  backend. Changes to shared operations generally belong in `burn-cubecl`.
- `burn-cubecl` implements Burn's backend operations, using local CubeCL kernels and reusable CubeK
  kernels.
- `burn-fusion` queues operations and coordinates fusion. `burn-cubecl-fusion` implements the fused
  kernels and optimizations, including integration with CubeK matmul and reduction kernels. Consider
  both fused and unfused execution when changing an operation.
- CubeK provides reusable kernel implementations built with CubeCL. CubeCL provides the kernel
  language, compilation, and runtime infrastructure. Some Burn kernels use CubeCL directly.
- Before adding a kernel, check existing Burn and CubeK implementations. Verify APIs against the
  dependency versions used by this checkout.

See the [backend chapter](contributor-book/src/project-architecture/backend.md#kernels-and-fusion)
for kernel integration, a matmul example, and guidance on where changes belong.

## Rust Code

- Use the simplest design that meets the requirements. Look for an existing abstraction before
  adding another representation of the same concept.
- Group code by responsibility and the state it owns. Prefer methods for operations on that state,
  and standard traits such as `From`, `TryFrom`, and `Display` where they fit.
- Keep configuration and serialized data separate from execution machinery.
- Represent meaningful cases with enums and exhaustive matches. Extract substantial branch bodies
  into named operations so the cases remain easy to read.
- Name types and methods for what they hold or do. Use newtypes when mixing up primitive values
  would be a bug, and named fields or builders when positional arguments obscure their meaning.
- Follow the surrounding module layout and visibility conventions. Keep responsibilities focused and
  public APIs small.
- Avoid unnecessary allocations and copies in hot paths. Reuse buffers where appropriate and measure
  performance changes with representative workloads.
- Document public behavior, preconditions, and failure modes. Keep comments concise and explain
  non-obvious decisions or invariants; avoid narrating the implementation or the history of a
  change.

## Validation

Start with checks and tests for the affected crates and backend. Follow the
[testing guide](contributor-book/src/getting-started/testing.md) for shared tensor and autodiff
suites, backend aliases, and fusion configurations. Include regression coverage for bug fixes.

The repository validation commands, run from the repository root, are:

```sh
cargo run-checks                       # Repository checks and tests with Flex
cargo run-checks --backend <backend>   # Select another backend for the tests
```

`cargo run-checks` is an alias for `cargo xtask validate`. It includes formatting, typo and
dependency checks, linting, feature and no-std checks, and backend tests. Inspect the resulting diff
because checks may modify files.

Report which checks actually ran and their results. Identify skipped checks and unavailable
hardware. Report failures, and verify their cause before claiming they are unrelated to the change.

## Pull Requests and AI Disclosure

Follow the [contribution policy](CONTRIBUTING.md#ai-assisted-contributions) when using AI tools. The
contributor remains responsible for understanding and validating the submitted changes.

- When preparing or updating a PR, read and use the [PR template](.github/pull_request_template.md).
- Complete its AI Usage section accurately. Describe your involvement in implementation, tests,
  investigation, documentation, the PR description, or review, including whether you assisted with
  specific tasks or carried out most or all of the work autonomously. Explicitly say when you wrote
  the description or created the PR. Do not omit your involvement.
- Describe only usage you know about. Do not guess another contributor's tool usage or claim that a
  human reviewed the changes unless they have confirmed it.
- Keep issue and PR descriptions succinct and focused on the problem and relevant evidence. For PRs,
  include the solution, validation, and relevant limitations; a few sentences plus test results are
  enough for a simple change.
- Edit generated prose for accuracy, relevance, and brevity before submitting it. Avoid large blocks
  of unedited AI prose, repeated summaries of the diff, and unrelated background. In code and review
  comments, give the shortest complete explanation needed rather than a large explanatory block.
- Keep AI disclosure in the PR description. Do not add AI coauthor or generated-by trailers to
  commit messages.
- When writing a review, identify it as agent-generated, give concrete findings with supporting
  evidence, and distinguish confirmed problems from questions or uncertainty. Maintainers make the
  final review and merge decisions.

Keep this file aligned with the code and contribution policy. Put detailed procedures in the
contributor documentation and link to them here.
