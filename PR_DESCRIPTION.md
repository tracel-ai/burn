* **Please check if the PR fulfills these requirements**
- [ ] The commit message follows our guidelines
- [ ] Docs have been added / updated (for bug fixes / features)


* **What kind of change does this PR introduce?** (Bug fix, feature, docs update, ...)

Bug fix + API change.

1. **`Dataset::get` now returns `Result<I, E>` instead of silently swallowing errors.**
   The `Dataset<I, E = DatasetError>` trait's `get` method now has the signature
   `fn get(&self, index: usize) -> Result<I, E>`, replacing the previous pattern where
   implementations (most notably `SqliteDataset`) `.unwrap()`'d internally and could panic
   with no context, or in some cases silently discarded real errors. `Err` is reserved for
   genuine retrieval failures on an in-bounds index (I/O errors, deserialization failures,
   corrupt rows, etc). Out-of-bounds indices still **panic**, matching `Vec`/slice indexing
   conventions, since a dataset is expected to know its own length.

   This propagated through every `Dataset` implementation in `burn-dataset` (in-memory,
   fake, dataframe, sqlite, vision, nlp, audio) and their transforms (composed, mapper,
   partial, shuffle, selection, sampler, window), plus `burn-core`'s dataloader, plus the
   custom `Dataset` implementations under `examples/` (`text-generation`, `simple-regression`,
   `text-classification`, `custom-csv-dataset` (both the CSV and dataframe backends), and
   `modern-lstm`), updating their `get` signatures and the two inference scripts that
   collected a `Dataset::iter()` into a plain `Vec<Item>`.

2. **Simplified the batching loop logic in `BatchDataloaderIterator::next`** (`crates/burn-core/src/data/dataloader/batch.rs`)
   now that `Dataset::get` has a real `Result`-based contract instead of the previous
   `Option`-based one.

3. **Fixed a class of silent, permanent hangs in the multi-threaded/multi-device orchestration code.**
   Several places in the training/data-loading pipeline wait for a *fixed count* of
   completion signals from worker threads (one `Done`/result per worker) before continuing.
   Previously, if a worker thread panicked (e.g. on a real error now surfaced by the
   `Dataset::get` change above) *before* sending its completion signal, the orchestrator
   would block on `recv()`/`join()` forever with no error ever surfaced to the user — just a
   panic backtrace printed to stderr from a background thread, easy to miss in a
   logged/non-interactive run. Fixed in three places by catching the panic on the worker
   side and explicitly propagating it as an error message/result instead of letting the
   worker die silently:
   - `MultiThreadDataLoader`'s worker pool (`crates/burn-core/src/data/dataloader/multithread.rs`):
     workers now catch panics from `Dataset::get` and send an explicit `Message::Error`,
     which the orchestrator turns into an immediate, clear panic instead of hanging on
     `Message::Done` counting. Covered by a new regression test that fails fast (via a
     timeout) instead of hanging CI if this regresses.
   - `MultiDevicesTrainStep` (`crates/burn-train/src/learner/supervised/step/train.rs`):
     per-device worker threads now catch panics from `TrainStep::step` and report them via
     a `WorkerMessage::Error`, instead of leaving the orchestrator's `recv()` loop blocked
     on an output that will never arrive.
   - `DdpTrainingStrategy::fit`'s worker join logic (`crates/burn-train/src/learner/supervised/strategies/ddp/strategy.rs`):
     replaced the sequential `for worker in secondary_workers { worker.join() }` /
     `main_handle.join()` pattern with a concurrent join — one small "reaper" thread per
     worker forwards its `.join()` result over a channel, and the orchestrator reacts to
     the first result (success or panic) to arrive, regardless of order. This matters
     specifically because sequential joins could block forever on a live-but-innocent peer
     thread that's itself permanently stuck waiting on the DDP collective-sync rendezvous
     for a *different* device that already died — meaning the actual panic might never even
     be reached by the old loop.


* **Does this PR introduce a breaking change?** (What changes might users need to make in their application due to this PR?)

Yes. `Dataset::get` signature changes from returning `Option<I>` (or similar ad-hoc
error handling per implementation) to `Result<I, E>`. Custom `Dataset` implementations
outside this crate will need to update their `get` method to return `Result<I, E>`
(defaulting `E = DatasetError` covers most cases), and out-of-bounds access must now
panic rather than return `None`. Callers iterating a `Dataset` via `.iter()` now get
`Result<I, E>` items instead of `I`.


* **Other information**:

**Known follow-up, not addressed in this PR**: the DDP strategy fix above only stops the
*orchestrator* thread from hanging silently — it does not stop a live-but-innocent peer
device thread from getting permanently stuck inside `DistributedSyncClient::submit_sync_collective`'s
`rx.recv()` (`crates/burn-backend/src/backend/distributed/client.rs`) waiting for a
collective-sync contribution from a device that already died. That stuck thread is
currently only reaped because the orchestrator's panic tears down the whole process; in a
context where this code isn't running on the process's main thread, it would leak forever
even after the error is reported.

The root cause is that all devices share clones of a single `mpsc::Sender` funneling into
one server thread (`DistributedSyncClient::new` / `DistributedSyncServer`,
`crates/burn-backend/src/backend/distributed/{client,server}.rs`), so the server has no
way to distinguish "a device is just slow" from "a device is dead" — `mpsc` only signals
disconnection once *every* sender clone is dropped, not a specific one.

A reasonably contained fix (no new channel topology needed) would be to reuse the existing
shared channel to carry an explicit failure signal: catch panics around
`learner.train_step(item)` in `DdpTrainEpoch::run`
(`crates/burn-train/src/learner/supervised/strategies/ddp/epoch.rs`), send a new
`ActionMessage::DeviceFailed(device_id)` before re-raising, and have the server respond by
sending an error through every currently-pending callback in `self.callbacks` instead of
leaving them parked — poisoning the server for the remainder of the run, since DDP
correctness requires aborting all replicas together rather than letting some continue with
mismatched gradients. Left out of this PR pending a decision on that poisoning semantics
and to keep this PR's scope to the `Dataset::get` contract change and the hangs it exposed.
