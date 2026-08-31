# Contributing to `burn-fusion`

## Error handling

Fusion reorders and merges work, so a failure in one kernel is not confined to one operation.
This is the policy that keeps that honest, and the invariants any change has to preserve.

### The rule

> A tensor reads back if and only if the work that was going to write it ran and succeeded.

Everything below follows from that one sentence. When you are unsure what a new path should do,
ask what this rule requires of it. It is also the oracle the property test in `stream/multi.rs`
checks over random interleavings — if you change the policy, that test is what tells you what you
broke.

### Vocabulary

A **claim** is the record that a tensor holds no data because the work meant to write it did not
get there. It is a [`TensorError`](../burn-ir/src/handle.rs): an `Arc<ExecutionError>` — the failing
work's own error, kept whole — plus a **depth** counting how many operations were skipped between
that failure and this tensor.

The cause is shared rather than reformatted, so propagating costs a refcount bump and identity is
pointer equality (`same_root`). A read below a long chain of skipped work still names the failure
that started it, and can tell that two tensors failed for the same reason. At depth zero a read
hands back that error itself, backtrace intact, so a caller can match on it.

A claim lives on the **tensor id**, in the `HandleContainer`, as `Handle::Errored` — a variant *of*
`Handle`, so recording a claim displaces the handle and frees the buffer. A claimed tensor has no
data, which is the honest state to be in.

### The scope

Every unit of work runs inside a [`WriteScope`](src/stream/execution/scope.rs). It is opened over
the work's IR, so its write set is exact, and it owns the three things each execution site used to
keep by hand:

- **the skip decision**, made once in the constructor, so skipping and entering can never
  interleave;
- **the catch**, so a unit that panics claims what it was going to write without any caller
  remembering to;
- **the claim** itself, on every failure path.

```rust
match WriteScope::over(ir, handles).run(OnPanic::Catch, |handles| op.execute(handles)) {
    Outcome::Ran => {}
    Outcome::Skipped => …,       // an input was claimed; the body never ran
    Outcome::Reported => …,      // it returned an error
    Outcome::Panicked(panic) => …, // it raised one, and the scope caught it
}
```

`OnPanic::Raise` is for work that already runs inside another scope — a fallback in the middle of a
fused block. It claims a reported failure exactly as `Catch` does, but lets a panic out, because
swallowing that one would let the block carry on as though the piece it could not serve had run.

**If you add an execution path, put it in a scope.** That is the whole of the policy; everything
below is what the scope is doing on your behalf.

### The four transitions

**Claim.** Work that fails records its error on every tensor it was going to write. It always
displaces: a handle registered *ahead* of the launch — as an in-place output is, aliased to its
input while the launch is still being planned — proves nothing about whether the kernel ever ran.

**Skip and propagate.** Before any unit of work runs, `input_error` asks whether a failure claims
any input. If one does, the work does not run, and its outputs take that same failure through
`propagated()` — same cause, one hop deeper. Running on unwritten bytes would turn a failure that
names one tensor into a wrong answer that names none.

**Recover.** Writing a claimed tensor clears the claim. Every `register_*` goes through the insert
that replaces the entry, so this holds however the write arrives. This is the property retrying
rests on: the attempt that fails claims the output it never wrote, the attempt that works writes it,
and nothing downstream is skipped.

**Release.** A claim lives exactly as long as the tensor carrying it — released by the tensor's
`Drop`, or by a `ReadWrite` read that consumes it (`take_error`). A drop raised while the thread is
unwinding cannot register then — that re-enters the client mid-unwind — so it is set aside and
replayed at the next call into the client on that thread, rather than dropped on the floor. Note that `input_error`
deliberately exempts `OperationIr::Drop`: a drop names its tensor as an input but does not read it,
and skipping drops would make claims outlive every tensor that could report them.

### Granularity — what counts as one unit of work

| Unit | Scope over | Claims on failure |
| --- | --- | --- |
| Unfused operation | that operation | its outputs |
| Fused block | every operation in the block | the block's whole write set |
| Fallback inside a block | that operation | its outputs |

Unfused operations get **one scope each, not one per segment**. A segment is just what happened to
be queued together, so a failure in one operation says nothing about the next unless they share a
tensor — and if they do, the next one skips on the claim its input now carries. Scoping the whole
loop would make an unrelated operation's outcome depend on queue order.

A fused block is one unit in both directions: one claimed input anywhere stops all of it, and a
failure anywhere leaves the whole write set unwritten. **This means fusion widens the blast radius
of a failure** — the same program reports a more precise cause with fusion off. That is a deliberate
consequence of a fused kernel being one kernel, not an oversight.

### Reporting versus raising

`Operation::execute` returns `Result<(), ExecutionError>`. **Prefer returning an error.** A reported
failure claims exactly what a raised one claims, but the claim carries the error whole — variant and
backtrace — where a panic payload is only a message. Raising still works, and still claims, because
the scope catches.

### Delivery

At the read, **as a value, never as a panic**: `take_error` first, then `Err(ExecutionError)` out of
`*_into_data`. A panic cannot take part in the release — it exits before the status branch that
decides whether to consume the claim — so delivering that way leaks the entry and leaves
`has_errors()` true for the life of the process.

`get_handle` still panics on a claimed tensor. That is a backstop for internal paths that do not ask
first, not the delivery mechanism. If you add a read path, call `take_error` and return the error.

A `to_device` of a claimed tensor produces a claimed tensor: `change_client_*` cannot return a
`Result`, so the destination takes the failure one hop down rather than the transfer panicking.

### Invariants to preserve

- **Free while clean.** Every check sits behind `has_errors()`, so nothing costs more than a
  predictable branch until something actually fails. `input_error` would otherwise allocate a boxed
  iterator per operation for an answer that is always `None`. Do not add an error check that pays
  before it needs to.
- **Bounded.** A claim is released with its tensor, so claims cannot accumulate. The container
  leaks claims only if the program leaks tensors. Any new path that takes ownership of a tensor id
  must release the claim on it.
- **Forward progress.** Consuming nothing after a failure is never safe: the policy re-plans the
  identical queue, re-selects the same strategy and fails the same way, without end. A plan that
  does not fit its segment is replaced before the walk begins, so it costs fusion rather than the
  work; `consume_stalled` remains as the guarantee of last resort.
- **`did_not_run` is load-bearing.** A drained operation that never ran was never replayed
  server-side, so the router's `free_handle` needs to know in order not to strand the buffer. If you
  add a path that skips work, record it. Work that cannot reach the execution directly — a
  `FallbackOp` outlives the borrow it was built from — records through the shared queue that
  `finish` merges.

### Known gaps

- Device failures in **unfused cubecl operations** do not enter this system. `ComputeClient::launch`
  returns `()`; the failure is a cubecl-side taint that surfaces at the read as `Err(ServerError)`.
  cubecl skips its own downstream work through its `ExecuteScope`, `burn-fusion` skips its own
  through `input_error`. The two layers stack rather than sharing one mechanism.
  `ComputeClient::check` is the seam if this is ever unified.
- The scope claims on failure rather than **on entry**. This was tried and measured, and rejected
  on both counts. Claiming the write set on the way in costs **+49 ns per operation, about 20%** of
  the execution path (249 → 298 ns/op on `execution_path_throughput`) — paid by every operation a
  program runs, on a crate whose purpose is reducing per-operation overhead. And the case for it is
  weaker than it looks: the headline argument was correctness under `panic = "abort"`, which is
  vacuous, because an abort ends the process and there is no later read for a claim to protect. What
  is genuinely lost is a check that an operation writes every output it declares — which, when
  implemented as a `debug_assert`, found nothing across the whole suite. The scope is still the API,
  so this can be revisited behind `over` and `run` without touching a call site if the second
  argument ever gets teeth.
- A drop set aside during an unwind is replayed by the next call into the client *on that thread*.
  A thread that panics and then never touches fusion again keeps those entries, and any claim on
  them, for the life of the process — the queue is thread-local and nothing else drains it. It is a
  far narrower leak than abandoning the registration outright, not the absence of one.
- Raising is still caught with `catch_unwind`, so a backend that panics rather than reporting relies
  on the unwinding panic runtime. Under `panic = "abort"` a panicking operation ends the process,
  which is a worse outcome than a claim but not one a claim could improve on — prefer reporting.
- The property test drives the unfused path. A fused block's granularity is pinned by targeted tests
  instead, because modelling where the fuser puts block boundaries would make the model a copy of
  the implementation.
