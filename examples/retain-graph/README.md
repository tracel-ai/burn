# retain-graph

Demonstrates `retain_graph` support in Burn — the ability to call `backward()` multiple times
on the same computational graph without destroying it, equivalent to PyTorch's
`loss.backward(retain_graph=True)`.

## Running the Example

```bash
# From the workspace root
cargo run -p retain-graph

# Or from the example directory
cd examples/retain-graph
cargo run
```

Expected output:

```
Gradient 1: Tensor { data: [5.0, 0.0], shape: [2], ... }
Gradient 2: Tensor { data: [0.0, 5.0], shape: [2], ... }
```

## What the Example Shows

The example builds a graph for the outer product `result = a ⊗ b` and then differentiates two
separate scalar slices of the result:

- `result1 = result[0, 0]` — equals `a[0] * b[0]`, so `∂/∂a = [b[0], 0] = [5, 0]`
- `result2 = result[1, 0]` — equals `a[1] * b[0]`, so `∂/∂a = [0, b[0]] = [0, 5]`

Because both backward passes originate from the same graph, the first call must not destroy it.
Uncommenting the `backward()` pair in `src/main.rs` demonstrates the original panic.

---

## Implementation: Adding `retain_graph` to Burn

### The Problem

Before this change, calling `backward()` on a tensor permanently destroyed the computational
graph. Every node was extracted from its `HashMap` by value (`HashMap::remove`) and consumed.
After one backward pass, nothing remained for a second.

The root cause was a chain of ownership transfers through the execution pipeline:

```
backward()
  steps.remove(&node_id)            // root step taken by value
  actions_builder.remove(&node_id)  // checkpoint builder consumed

BreadthFirstSearch::traverse()
  steps.remove(&id)                 // every visited step removed and owned
  callback(step)                    // step moved into tape

execute_steps()
  tape.into_iter()                  // tape consumed
  step.step(grads, checkpointer)    // Step::step(self: Box<Self>) destroys step
```

### Ownership Bottlenecks Identified

Five interconnected bottlenecks made the graph non-reusable:

1. **`Step::step(self: Box<Self>)`** — the trait method consumed the boxed step on execution
2. **`Backward::backward(self, ops)`** — the backward handler consumed itself
3. **`HashMap::remove()` in traversal** — each step was extracted and could not be re-visited
4. **`actions_builder.remove()` in `build_tape`** — checkpoint builders removed per node
5. **`Box<dyn Any + Send>` for checkpoint state** — not cloneable, so builders could not be reused

### What Was Changed and Why

#### 1. `Step` trait — `crates/burn-autodiff/src/graph/base.rs`

**Before:**
```rust
fn step(self: Box<Self>, grads: &mut Gradients, checkpointer: &mut Checkpointer);
```

**After:**
```rust
fn step(&self, grads: &mut Gradients, checkpointer: &mut Checkpointer);
```

**Why:** The consuming `Box<Self>` receiver made it impossible to call `step` twice on the
same object. By borrowing `&self`, the step can execute any number of times while remaining
in the `HashMap`.

**Breaking change:** Any code outside `burn-autodiff` that implements the `Step` trait must
update this method signature.

#### 2. `OpsStep::step` — `crates/burn-autodiff/src/ops/base.rs`

**Before:**
```rust
fn step(self: Box<Self>, grads: &mut Gradients, checkpointer: &mut Checkpointer) {
    self.backward.backward(self.ops, grads, checkpointer);
}
```

**After:**
```rust
fn step(&self, grads: &mut Gradients, checkpointer: &mut Checkpointer) {
    let ops = Ops::new(
        self.ops.parents.clone(),
        self.ops.node.clone(),
        self.ops.state.clone(),  // S: Clone was already required
    );
    self.backward.backward(ops, grads, checkpointer);
}
```

**Why:** `backward` now takes owned `ops`, but the step is borrowed. The ops are reconstructed
by cloning. `S: Clone` was already a bound on `Backward::State`, so this is always valid. For
tensor primitives the clone is a cheap `Arc` reference-count increment, not a data copy.

#### 3. `Backward` trait — `crates/burn-autodiff/src/ops/backward.rs`

**Before:**
```rust
fn backward(self, ops: Ops<Self::State, N>, grads: &mut Gradients, checkpointer: &mut Checkpointer);
```

**After:**
```rust
fn backward(&self, ops: Ops<Self::State, N>, grads: &mut Gradients, checkpointer: &mut Checkpointer);
```

**Why:** All ~91 implementations of this trait across `tensor.rs`, `module.rs`,
`activation.rs`, `maxmin.rs`, and `sort.rs` are **zero-sized types (ZSTs)** with no fields.
The Burn documentation explicitly states "Concrete types implementing this trait should not
have any state." None of them reference `self` in their bodies. Changing `self` to `&self`
is therefore purely mechanical with no semantic effect.

**Breaking change:** Any code outside `burn-autodiff` that implements the `Backward` trait
must update this method signature.

`CatStep` was a special case: it implements `Step` directly (not `Backward`) and previously
consumed `self.nodes.into_iter()`. Fixed by switching to `self.nodes.iter()` with `.cloned()`
and using `self.nodes.iter().zip(self.dim_sizes.iter().copied())`.

#### 4. Checkpoint state — `crates/burn-autodiff/src/checkpoint/state.rs` and `builder.rs`

**Before:**
```rust
pub(crate) type StateContent = Box<dyn Any + Send>;
```

**After:**
```rust
pub(crate) type StateContent = Arc<dyn Any + Send>;
```

**Why:** `CheckpointingAction::Computed` stores a snapshot of a tensor primitive. With `Box`,
the snapshot was single-use and could not be cloned. With `Arc`, the same snapshot can be
cheaply shared across multiple backward passes via reference-count increments.

This change also enables `CheckpointingAction` and `CheckpointerBuilder` to implement `Clone`,
which is required by the retain path when borrowing checkpoint builders from the map.

A new `extend_ref` method was added to `CheckpointerBuilder`:

```rust
pub(crate) fn extend_ref(&mut self, other: &CheckpointerBuilder) {
    for other_action in other.explicit_actions.iter() {
        self.explicit_actions.push(other_action.clone())
    }
    for other_unsure in other.backup_actions.iter() {
        self.backup_actions.push(other_unsure.clone())
    }
}
```

The existing `extend` method (taking ownership) is unchanged and used by the non-retain path.

**Breaking change:** Any code that constructs `CheckpointingAction::Computed { state_content }`
directly must now supply an `Arc<dyn Any + Send>` instead of a `Box<dyn Any + Send>`.

#### 5. Non-destructive traversal — `crates/burn-autodiff/src/graph/traversal.rs`

A new `traverse_retaining` method was added alongside the existing `traverse`:

```rust
pub fn traverse_retaining<F, I>(
    &self,
    root_id: NodeId,
    steps: &HashMap<NodeId, I>,   // immutable borrow — no removal
    mut callback: F,
) where
    F: FnMut(NodeId, &I),         // callback receives reference, not owned value
    I: TraversalItem,
```

The existing `traverse` (which uses `remove()`) is unchanged and still used by the normal
`backward()` path.

**Why:** BFS traversal with `get()` instead of `remove()` leaves every step in the `HashMap`
intact. The callback receives a reference, so no node is consumed.

#### 6. Retain path in the server — `crates/burn-autodiff/src/runtime/server.rs`

Three new methods were added to `AutodiffServer`:

```rust
pub fn backward_retain(&mut self, grads: Gradients, node_id: NodeId) -> Gradients {
    let (tape, checkpointer) = self.build_tape_retaining(node_id);
    Self::execute_steps_retaining(tape, &self.steps, grads, checkpointer)
    // No cleanup — graph is deliberately preserved
}
```

- **`build_tape_retaining`** builds a `Vec<Vec<NodeId>>` (IDs only, not owned steps) by
  borrowing steps and checkpoint builders via `extend_ref`. No nodes are consumed.
- **`execute_steps_retaining`** looks up each step by `NodeId` from the (intact) `steps` map
  and calls `step.step(...)` on a borrowed reference.

The key difference from `build_tape` / `execute_steps` is that the tape stores `NodeId`
values rather than `StepBoxed` values, so the steps remain in the server's `HashMap` throughout.

#### 7. Client and graph layers — `client.rs`, `graph.rs`

`AutodiffClient` trait gained a new required method:

```rust
fn backward_retain<B: Backend>(&self, tensor: &AutodiffTensor<B>) -> Gradients;
```

`GraphMutexClient::backward_retain` intentionally omits the
`GraphCleaner::cleanup_orphaned_entries()` call that follows the normal `backward()` path,
because cleaning up graph entries would destroy the retained graph.

**Breaking change:** Any code outside `burn-autodiff` that implements `AutodiffClient` must
now implement `backward_retain`.

#### 8. Public API surface

The method was threaded through the full call stack:

| Layer | Change |
|---|---|
| `AutodiffClient` trait (`client.rs`) | Added `backward_retain<B>(&self, tensor: &AutodiffTensor<B>) -> Gradients` |
| `GraphMutexClient` (`graph.rs`) | Implements `backward_retain` — no orphan cleanup |
| `AutodiffTensor<B>` (`tensor.rs`) | Added `pub fn backward_retain(&self) -> Gradients` |
| `AutodiffBackend` trait (`burn-backend/src/backend/base.rs`) | Added `fn backward_retain(tensor: &FloatTensor<Self>) -> Self::Gradients` |
| `Autodiff<B, C>` (`burn-autodiff/src/backend.rs`) | Implements `backward_retain` by delegating to `tensor.backward_retain()` |
| `Dispatch` backend (`burn-dispatch/src/backend.rs`) | Implements `backward_retain` for all hardware variants via `as_autodiff().backward_retain()` |
| `Tensor<D>` (`burn-tensor/src/tensor/api/autodiff.rs`) | Added `pub fn backward_retain(&self) -> B::Gradients` |

**Breaking change:** Any code that implements `AutodiffBackend` must now implement
`backward_retain`.

### Why Cloning Was Not the Central Solution

The insight here: **cloning was only needed because ownership was being extracted**. By keeping
steps in the `HashMap` and borrowing them instead, no clone of `Box<dyn Step>` (impossible
without a `clone_box` method) was ever needed.

The only clones introduced:
- `ops.state.clone()` in `OpsStep::step` — `S: Clone` was already required
- `Arc<dyn Any + Send>` clone in checkpoint builder — an `Arc` reference-count bump, not a data copy
- `Arc<dyn RetroForward>` clone — already an `Arc`, trivially cheap

### Performance Impact

The non-retain `backward()` path is now marginally slower: `OpsStep::step` clones `ops.state`
on every step (previously moved). In practice this is negligible because tensor primitives are
`Arc`-backed — the clone is an atomic increment, not a buffer copy. No regressions were observed
in the existing test suite.

---

## Tests

Three tests were added in `crates/burn-backend-tests/tests/autodiff/retain_graph.rs`:

1. **`should_produce_same_gradients_on_repeated_backward`** — two retain passes on the same
   graph yield identical gradients
2. **`should_match_standard_backward_gradients`** — retain gradients numerically match those
   from a standard consuming backward pass
3. **`should_allow_multiple_backward_after_retain`** — retain passes followed by a final
   consuming `backward()` all produce consistent gradients

Tests run against both `NoCheckpointing` (default) and `BalancedCheckpointing` strategies
via the existing test macro infrastructure.

All 1763 existing tests continue to pass with no regressions.

---

## Deviations from Burn Contribution Guidelines

The following notes are relevant for the pull request review.

### vs. "Adding a New Operation to Burn"

The contribution guide for new operations (e.g. `pow`) describes adding a tensor compute
operation that every backend must implement. `backward_retain` is not a compute operation — it
is autodiff infrastructure. The following intentional deviations apply:

| Guideline | Status | Reason |
|---|---|---|
| Add op to `FloatTensorOps` / `IntTensorOps` | Not done | `backward_retain` is not a forward compute op |
| Add quantized tensor counterpart (`q_*`) | Not done | Quantized inference does not perform autodiff backward |
| Implement for each compute backend (NdArray, Wgpu, Tch, …) | Not done | Handled entirely at the autodiff layer; compute backends are not involved |
| Add to burn-fusion / burn-ir stream | Not done | Fusion is for forward ops; backward is not fused |
| Tests under `tests/tensor/float/ops/` | Tests placed under `tests/autodiff/` instead | These are autodiff-layer tests, not backend compute tests |

What was followed correctly: the op was added to the public `Tensor<D>` API, added as a trait
method on `AutodiffBackend`, threaded through `burn-dispatch`, and tested in
`burn-backend-tests`.

### vs. "Submitting Examples to Burn"

| Guideline | Status | Notes |
|---|---|---|
| `readme = "README.md"` in `Cargo.toml` | **Missing** | Must be added to `examples/retain-graph/Cargo.toml` |
| `publish = false` in `Cargo.toml` | **Missing** | Should be added to prevent accidental publishing |
| `edition.workspace = true` / `version.workspace = true` | **Missing** | Hard-coded `edition = "2021"` instead of inheriting from workspace |
| `[lints] workspace = true` | **Missing** | Workspace lint configuration not applied |
| Library crate with `examples/<name>.rs` entry point | Not followed | Example uses a binary crate (`src/main.rs`) and runs with `cargo run -p retain-graph` rather than `cargo run --example retain-graph` |

The `Cargo.toml` should be updated to at minimum:

```toml
[package]
name = "retain-graph"
version.workspace = true
edition.workspace = true
publish = false
readme = "README.md"

[lints]
workspace = true

[dependencies]
burn = { path = "../../crates/burn", features = ["autodiff", "ndarray"] }
```
