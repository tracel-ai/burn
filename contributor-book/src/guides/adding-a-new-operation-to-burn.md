# Adding a New Operation to Burn

## Choosing where the operation belongs

First consider the operation's intended users and scope:

- General-purpose tensor operations shared across domains may belong in Burn's core tensor API.
- Domain-specific operations belong in the corresponding extension crate, such as
  [`burn-vision`](https://github.com/tracel-ai/burn/tree/main/crates/burn-vision),
  [`burn-linalg`](https://github.com/tracel-ai/burn/tree/main/crates/burn-linalg), or
  [`burn-signal`](https://github.com/tracel-ai/burn/tree/main/crates/burn-signal). These crates keep
  specialized APIs and their backend implementations together, with capabilities enabled as needed.
- Application-specific or experimental operations can live in your application or a separate crate.
  They do not need to be upstreamed to be used with Burn.

Then decide how to implement the operation. A composition of existing tensor operations can be
exposed as a function or extension trait in any of these locations without changing the backend
contract. If custom kernels are needed, a
[backend extension](https://burn.dev/books/burn/advanced/backend-extension/) lets you define the
operation and its backend implementations in an extension crate, including one maintained outside
Burn.

The sections below describe adding an operation to the core tensor API and, when a new primitive is
needed, its backend contract and routing. For domain or external extensions, follow the backend
extension guide and the conventions of the crate that owns the operation.

## Public tensor API and bridge

Add the method to the appropriate file in `crates/burn-tensor/src/tensor/api`: `base.rs` for common
operations, `numeric.rs` for shared numeric operations, or `float.rs`, `int.rs`, and `bool.rs` for
kind-specific operations. Neural-network operations and activations also have function APIs.
Document shapes, broadcasting, dtype behavior, examples, and runtime preconditions. Add necessary
validation in the tensor checks.

The public type is `Tensor<D, K>`, with an opaque `BridgeTensor` primitive. Keep generic method
bodies thin: route through a non-generic `*_impl` helper where needed and the corresponding kind
operations in `crates/burn-tensor/src/bridge/ops`. Do not expose backend types in ordinary public
method signatures. See [Tensor Architecture](../project-architecture/tensor.md).

## Backend contract and dispatch

Define the primitive operation in the relevant trait under `crates/burn-backend/src/backend/ops`.
Shared names are prefixed by kind, such as `float_powf` and `int_powf`. A default implementation may
compose existing primitive operations where appropriate. Dtypes are runtime values; there are no
backend-associated float or integer element types.

Add forwarding in `crates/burn-dispatch/src/ops`. Built-in implementations use
`#[backend_dispatch]`, which selects the runtime backend and handles autodiff contexts. Operations
needing custom routing can use `#[backend_dispatch(skip)]` and an explicit implementation; follow an
existing operation with matching inputs and outputs.

For a complete existing path, trace `Tensor::powf` through the numeric bridge,
`Dispatch::float_powf`, and `FloatTensorOps::float_powf`.

## Concrete backends and decorators

Implement the operation on the supported concrete backends. For CubeCL kernels, the implementation
is on `CubeBackend`; runtime-specific execution is selected by its device. Handle supported dtypes
and layouts, including non-contiguous inputs where the operation permits them.

A new primitive also needs the applicable decorator and graph paths:

- **Autodiff:** implement the derivative in `crates/burn-autodiff/src/ops`. Follow neighboring
  operations for `Backward`, saved state, checkpointing, broadcasting reductions, and tracked versus
  untracked inputs. The forward pass must save only the state needed by the backward computation.
- **IR:** add the representation under `crates/burn-ir/src/operation.rs` when the operation is
  recorded or transmitted, including shape and scalar arguments.
- **Fusion:** record the operation in `crates/burn-fusion/src/ops`. Kernel fusion support, when
  appropriate, also involves `burn-cubecl-fusion`; merely recording an operation does not make it
  fusible with its neighbors.
- **Router:** record the operation in `crates/burn-router/src/ops` and add its execution to
  `TensorInterpreter` in `crates/burn-router/src/interpreter.rs`. The recorded IR and interpreter
  must agree on the operation's inputs, outputs, and metadata.
- **Remote and capture:** verify the operation through these consumers of the router layer.
  `burn-remote` uses `BackendRouter<RemoteChannel>` and executes received operations through
  `TensorInterpreter`; `burn-capture` uses `BackendRouter<CaptureChannel>` to record them without
  execution. Ordinary operation support belongs in the shared router layer; changes in these crates
  are needed when the operation requires additional transport or capture handling.

Some operations have intentional backend limitations. Make them explicit in documentation and errors
rather than assuming every runtime has the same capability.

## Tests and documentation

Add forward and backward coverage to `burn-backend-tests`; see the
[testing guide](../getting-started/testing.md). Verify shapes, values, broadcasting, and dtypes,
plus empty inputs or non-contiguous layouts when relevant. Check every differentiable input and any
saved state used by the backward pass.

Run the affected suites with the target backend, and run the repository validation workflow before
submitting. Update the Burn Book if the operation adds a new user workflow or changes existing
semantics. Keep examples aligned with the runtime device API.
