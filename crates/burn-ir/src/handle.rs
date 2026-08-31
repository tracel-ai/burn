use alloc::string::String;
use alloc::sync::Arc;
use hashbrown::HashMap;

use crate::{BackendIr, TensorHandle, TensorId, TensorIr, TensorStatus};

/// Keep all [tensor handles](BackendIr::Handle) in one place and ensure that all resources
/// are used optimally.
pub struct HandleContainer<H> {
    handles: HashMap<TensorId, Handle<H>>,
    counter: u64,
    /// How many entries are [`Handle::Errored`].
    ///
    /// Kept so the checks a claim makes necessary cost nothing while nothing
    /// has failed, which is nearly always: reading it is a branch, where
    /// asking the question properly is a hash lookup per handle fetch and a
    /// boxed iterator per operation. Maintained only by [`put`](Self::put)
    /// and [`take`](Self::take) — every mutation goes through those two, so
    /// it cannot drift from the map.
    claimed: usize,
}

// Hand-written perfect derive as we don't require `H: Default`.
impl<H> Default for HandleContainer<H> {
    fn default() -> Self {
        Self {
            handles: HashMap::new(),
            counter: 0,
            claimed: 0,
        }
    }
}

impl<H: Clone> HandleContainer<H> {
    /// Fork the container, useful for autotune.
    pub fn fork(&self) -> Self {
        let mut handles = HashMap::with_capacity(self.handles.len());

        for (id, handle) in self.handles.iter() {
            handles.insert(*id, handle.clone());
        }

        Self {
            handles,
            counter: self.counter,
            claimed: self.claimed,
        }
    }
}

impl<H> core::fmt::Debug for HandleContainer<H> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HandleContainer")
            .field("handles", &self.handles.keys()) // only care about the IDs when debugging
            .field("counter", &self.counter)
            .finish()
    }
}

/// Why a tensor holds no data: the work that was going to write it did not
/// run, so its bytes were never produced.
///
/// The root message is behind an [`Arc`] shared by every tensor one failure
/// claims, so propagating a failure downstream costs a refcount bump and two
/// tensors below the same root report the same thing. Identity is pointer
/// equality on that root — see [`same_failure`](Self::same_failure).
#[derive(Clone)]
pub struct TensorError {
    failure: Arc<Failure>,
    /// How many operations were skipped between the failure and this tensor.
    /// Zero for the outputs of the work that actually failed.
    depth: u32,
}

struct Failure {
    /// What the failing work reported.
    root: String,
}

impl TensorError {
    /// A fresh failure, claiming the tensors the work that raised `root` was
    /// going to write.
    pub fn new(root: impl Into<String>) -> Self {
        Self {
            failure: Arc::new(Failure { root: root.into() }),
            depth: 0,
        }
    }

    /// The same failure, one operation further downstream — for the outputs of
    /// work that was skipped because an input carried this error.
    ///
    /// The root is shared rather than reformatted, so a read below a long
    /// chain of skips still names the failure that started it.
    pub fn propagated(&self) -> Self {
        Self {
            failure: self.failure.clone(),
            depth: self.depth.saturating_add(1),
        }
    }

    /// What the failing work reported.
    pub fn root(&self) -> &str {
        &self.failure.root
    }

    /// How many operations were skipped between the failure and this tensor.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Whether both tensors were claimed by the same failure, however far
    /// downstream each one is.
    pub fn same_failure(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.failure, &other.failure)
    }
}

impl core::fmt::Display for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.depth {
            0 => write!(f, "the work writing it failed: {}", self.failure.root),
            skipped => write!(
                f,
                "the work writing it was skipped {skipped} operation(s) below a failure: {}",
                self.failure.root
            ),
        }
    }
}

impl core::fmt::Debug for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorError")
            .field("root", &self.failure.root)
            .field("depth", &self.depth)
            .finish()
    }
}

/// Backend [tensor handle](BackendIr::Handle) wrapper tracking their creation state
#[derive(Clone)]
pub enum Handle<H> {
    /// No [tensor handle](BackendIr::Handle) has been created yet
    NotInit,
    /// A [tensor handle](BackendIr::Handle) has been created
    Existing(H),
    /// No handle will be created: the work that was going to write this
    /// tensor did not run.
    ///
    /// The fact lives on the tensor rather than on whatever queue was
    /// executing, so work that shares no tensor with the failure is
    /// unaffected, and the error reaches a caller only when one of *these*
    /// tensors is read. The entry is released by the tensor's own `Drop` like
    /// any other, which is what bounds the set to the tensors still alive.
    Errored(TensorError),
}

impl<H: Clone> HandleContainer<H> {
    /// Create a new HandleContainer
    pub fn new() -> Self {
        Self {
            handles: HashMap::new(),
            counter: 0,
            claimed: 0,
        }
    }

    /// Insert `handle`, keeping [`claimed`](Self#structfield.claimed) in step.
    fn put(&mut self, id: TensorId, handle: Handle<H>) -> Option<Handle<H>> {
        if let Handle::Errored(_) = handle {
            self.claimed += 1;
        }
        let previous = self.handles.insert(id, handle);
        if let Some(Handle::Errored(_)) = previous {
            self.claimed -= 1;
        }
        previous
    }

    /// Remove `id`'s entry, keeping [`claimed`](Self#structfield.claimed) in
    /// step.
    fn take(&mut self, id: &TensorId) -> Option<(TensorId, Handle<H>)> {
        let entry = self.handles.remove_entry(id);
        if let Some((_, Handle::Errored(_))) = &entry {
            self.claimed -= 1;
        }
        entry
    }

    /// Whether any tensor is claimed by a failure.
    ///
    /// A branch that lets the claim checks cost nothing while nothing has
    /// failed. False means no [`error`](Self::error) lookup can find
    /// anything, so the caller can skip asking — worth a branch where asking
    /// costs a boxed iterator over an operation's inputs.
    pub fn has_claims(&self) -> bool {
        self.claimed > 0
    }

    /// Register a handle for the given [tensor id](TensorId).
    pub fn register_handle(&mut self, id: TensorId, handle: H) {
        self.put(id, Handle::Existing(handle));
    }

    /// Whether a usable handle exists.
    ///
    /// False for a tensor claimed by a failure ([`Handle::Errored`]): the
    /// entry is there, but there is no data behind it.
    pub fn has_handle(&self, id: &TensorId) -> bool {
        matches!(self.handles.get(id), Some(Handle::Existing(_)))
    }

    /// Get the reference to a handle.
    ///
    /// `None` for a tensor claimed by a failure, the same as one that was
    /// never produced — the error is delivered by [`get_handle`](Self::get_handle),
    /// at the read, and asked for ahead of time with [`error`](Self::error).
    pub fn get_handle_ref(&self, id: &TensorId) -> Option<&H> {
        match self.handles.get(id) {
            Some(Handle::Existing(handle)) => Some(handle),
            _ => None,
        }
    }

    /// Mark `id` as claimed by `error`: the work that was going to write it
    /// did not run, so a read of it must fail rather than hand back whatever
    /// the tensor id resolves to.
    ///
    /// Overwrites a handle that is already there, which is not the same as
    /// losing data: a handle can be registered *before* the work that fills
    /// it, and an in-place output is registered as an alias of its input
    /// while planning the launch. A buffer that exists is not a buffer that
    /// was written, so only the work reaching its end proves the bytes are
    /// there.
    ///
    /// For `id`s that are a known write set. To claim a set that may include
    /// tensors other work already wrote, use
    /// [`claim_unwritten`](Self::claim_unwritten).
    pub fn claim(&mut self, id: TensorId, error: TensorError) {
        self.put(id, Handle::Errored(error));
    }

    /// [`claim`](Self::claim), but leaving alone any tensor that already has
    /// a handle.
    ///
    /// For claiming broadly — a failure that cannot say which work it came
    /// from — where the set is known to include tensors other work wrote and
    /// clobbering them would turn one failure into several.
    pub fn claim_unwritten(&mut self, id: TensorId, error: TensorError) {
        if let Some(Handle::Existing(_)) = self.handles.get(&id) {
            return;
        }
        self.put(id, Handle::Errored(error));
    }

    /// The failure claiming `id`, if one does.
    pub fn error(&self, id: &TensorId) -> Option<&TensorError> {
        match self.handles.get(id) {
            Some(Handle::Errored(error)) => Some(error),
            _ => None,
        }
    }

    /// Get the handle for the given [tensor id](TensorId). The status is used to determine if the
    /// tensor should be popped out of the current tensor map, necessary for inplace operations.
    ///
    /// # Warnings
    ///
    /// Make sure the status corresponds to the operation you want to execute the handle on,
    /// otherwise you might remove a tensor handle that will be required in the future.
    pub fn get_handle(&mut self, id: &TensorId, status: &TensorStatus) -> H {
        // Checked before the entry is taken: an unwind past a `remove_entry`
        // would clear the very claim that explains it, and the next read of
        // the same tensor would fail on a bare missing handle instead.
        //
        // Behind `has_claims` so the common path keeps its single lookup:
        // this is the hottest read in the system, and nothing is claimed
        // unless something has actually failed.
        if self.has_claims()
            && let Some(Handle::Errored(error)) = self.handles.get(id)
        {
            panic!("Tensor {id:?} was never written: {error}");
        }

        let (id, handle) = self
            .take(id)
            .unwrap_or_else(|| panic!("Should have handle for tensor {id:?}"));

        match handle {
            Handle::Existing(handle) => match status {
                TensorStatus::ReadOnly => {
                    self.put(id, Handle::Existing(handle.clone()));
                    handle
                }
                TensorStatus::ReadWrite => handle,
                TensorStatus::NotInit => panic!(
                    "Cannot get uninitialized tensor {id:?}. Tensor exist but with wrong status"
                ),
            },
            Handle::NotInit => panic!("Cannot get uninitialized handle {id:?}."),
            // Unreachable: the claim is checked above, before the entry is
            // taken.
            Handle::Errored(error) => panic!("Tensor {id:?} was never written: {error}"),
        }
    }

    /// Get the tensor handle for the given [tensor intermediate representation](TensorIr).
    pub fn get_tensor_handle(&mut self, tensor: &TensorIr) -> TensorHandle<H> {
        TensorHandle {
            handle: self.get_handle(&tensor.id, &tensor.status),
            shape: tensor.shape.clone(),
        }
    }

    /// Get the [float tensor](burn_backend::backend::BackendTypes::FloatTensorPrimitive) corresponding to the
    /// given [tensor intermediate representation](TensorIr).
    pub fn get_float_tensor<B>(&mut self, tensor: &TensorIr) -> B::FloatTensorPrimitive
    where
        B: BackendIr<Handle = H>,
    {
        B::float_tensor(self.get_tensor_handle(tensor))
    }

    /// Get the [int tensor](burn_backend::backend::BackendTypes::IntTensorPrimitive) corresponding to the
    /// given [tensor intermediate representation](TensorIr).
    pub fn get_int_tensor<B>(&mut self, tensor: &TensorIr) -> B::IntTensorPrimitive
    where
        B: BackendIr<Handle = H>,
    {
        B::int_tensor(self.get_tensor_handle(tensor))
    }

    /// Get the [bool tensor](burn_backend::backend::BackendTypes::BoolTensorPrimitive) corresponding to the
    /// given [tensor intermediate representation](TensorIr).
    pub fn get_bool_tensor<B>(&mut self, tensor: &TensorIr) -> B::BoolTensorPrimitive
    where
        B: BackendIr<Handle = H>,
    {
        B::bool_tensor(self.get_tensor_handle(tensor))
    }

    /// Get the [quantized tensor](burn_backend::backend::BackendTypes::QuantizedTensorPrimitive) corresponding to the
    /// given [tensor intermediate representation](TensorIr).
    pub fn get_quantized_tensor<B>(&mut self, tensor: &TensorIr) -> B::QuantizedTensorPrimitive
    where
        B: BackendIr<Handle = H>,
    {
        B::quantized_tensor(self.get_tensor_handle(tensor))
    }

    /// Register a new [float tensor](burn_backend::backend::BackendTypes::FloatTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_float_tensor<B>(&mut self, id: &TensorId, tensor: B::FloatTensorPrimitive)
    where
        B: BackendIr<Handle = H>,
    {
        let handle = B::float_tensor_handle(tensor);
        self.put(*id, Handle::Existing(handle));
    }

    /// Register a new [quantized tensor](burn_backend::backend::BackendTypes::QuantizedTensorPrimitive) with the corresponding [tensor ids](TensorId).
    pub fn register_quantized_tensor<B>(
        &mut self,
        id: &TensorId,
        tensor: B::QuantizedTensorPrimitive,
    ) where
        B: BackendIr<Handle = H>,
    {
        let handle = B::quantized_tensor_handle(tensor);
        self.put(*id, Handle::Existing(handle));
    }

    /// Register a new [int tensor](burn_backend::backend::BackendTypes::IntTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_int_tensor<B>(&mut self, id: &TensorId, tensor: B::IntTensorPrimitive)
    where
        B: BackendIr<Handle = H>,
    {
        let handle = B::int_tensor_handle(tensor);
        self.put(*id, Handle::Existing(handle));
    }

    /// Register a new [bool tensor](burn_backend::backend::BackendTypes::BoolTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_bool_tensor<B>(&mut self, id: &TensorId, tensor: B::BoolTensorPrimitive)
    where
        B: BackendIr<Handle = H>,
    {
        let handle = B::bool_tensor_handle(tensor);
        self.put(*id, Handle::Existing(handle));
    }

    /// Remove tensor handle from container.
    pub fn remove_handle(&mut self, id: TensorId) -> Option<Handle<H>> {
        self.take(&id).map(|(_, handle)| handle)
    }

    /// Remove tensor handle from container if writable
    pub fn free(&mut self, tensor: &TensorIr) {
        match tensor.status {
            TensorStatus::ReadOnly => (),
            TensorStatus::NotInit => (),
            TensorStatus::ReadWrite => {
                self.take(&tensor.id);
            }
        };
    }

    /// Returns the number of handles.
    pub fn num_handles(&self) -> usize {
        self.handles.len()
    }

    /// Returns the IDs of all currently registered handles.
    ///
    /// Useful for snapshotting which handles exist at a point in time (e.g., before
    /// executing on a forked context) so that newly registered output handles can
    /// be detected afterwards.
    pub fn handle_ids(&self) -> impl Iterator<Item = &'_ TensorId> {
        self.handles.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorId;

    /// Helper to create a TensorId for tests.
    fn tid(value: u64) -> TensorId {
        TensorId::new(value)
    }

    #[test]
    fn fork_clones_existing_handles() {
        let mut container = HandleContainer::<String>::new();
        container.register_handle(tid(1), "input_a".to_string());
        container.register_handle(tid(2), "input_b".to_string());

        let fork = container.fork();

        assert_eq!(fork.num_handles(), 2);
        assert!(fork.get_handle_ref(&tid(1)).is_some());
        assert!(fork.get_handle_ref(&tid(2)).is_some());
    }

    #[test]
    fn fork_is_isolated_from_original() {
        // This test documents the core of the autotune clone bug:
        // output handles registered in a fork do NOT appear in the original.
        let mut container = HandleContainer::<String>::new();
        container.register_handle(tid(1), "input_a".to_string());

        let mut fork = container.fork();

        // Simulate an optimization registering output handles in the fork.
        fork.register_handle(tid(100), "output_x".to_string());
        fork.register_handle(tid(101), "output_y".to_string());

        // The fork has the output handles.
        assert_eq!(fork.num_handles(), 3);
        assert!(fork.get_handle_ref(&tid(100)).is_some());
        assert!(fork.get_handle_ref(&tid(101)).is_some());

        // But the original does NOT — these output handles are lost.
        assert_eq!(container.num_handles(), 1);
        assert!(container.get_handle_ref(&tid(100)).is_none());
        assert!(container.get_handle_ref(&tid(101)).is_none());
    }

    #[test]
    fn fork_mutations_do_not_affect_original() {
        let mut container = HandleContainer::<String>::new();
        container.register_handle(tid(1), "original_value".to_string());

        let mut fork = container.fork();

        // Overwrite a handle in the fork (e.g., inplace output reuse).
        fork.register_handle(tid(1), "modified_in_fork".to_string());

        // Original is unchanged.
        assert_eq!(
            container.get_handle_ref(&tid(1)),
            Some(&"original_value".to_string())
        );
        assert_eq!(
            fork.get_handle_ref(&tid(1)),
            Some(&"modified_in_fork".to_string())
        );
    }

    /// A claimed tensor has no usable handle: the read must fail rather than
    /// hand back whatever the id resolves to.
    #[test]
    fn a_claimed_tensor_has_no_handle() {
        let mut container = HandleContainer::<String>::new();
        container.claim(tid(1), TensorError::new("the kernel failed to compile"));

        assert!(!container.has_handle(&tid(1)));
        assert!(container.get_handle_ref(&tid(1)).is_none());
        assert_eq!(
            container.error(&tid(1)).map(|error| error.root()),
            Some("the kernel failed to compile"),
        );
    }

    /// A broad claim leaves alone what other work already wrote, so one
    /// failure does not turn into several.
    #[test]
    fn a_broad_claim_never_displaces_data() {
        let mut container = HandleContainer::<String>::new();
        container.register_handle(tid(1), "written".to_string());
        container.claim_unwritten(tid(1), TensorError::new("something else failed"));

        assert_eq!(
            container.get_handle_ref(&tid(1)),
            Some(&"written".to_string())
        );
        assert!(container.error(&tid(1)).is_none());
    }

    /// A claim on a known write set overwrites: a handle can be registered
    /// before the work that fills it — an in-place output is aliased to its
    /// input while the launch is still being planned — so a handle that
    /// exists is not a buffer that was written.
    #[test]
    fn claiming_a_write_set_overwrites_a_handle_registered_ahead_of_the_work() {
        let mut container = HandleContainer::<String>::new();
        container.register_handle(tid(1), "aliased_input_buffer".to_string());
        container.claim(tid(1), TensorError::new("the launch failed"));

        assert!(container.get_handle_ref(&tid(1)).is_none());
        assert_eq!(
            container.error(&tid(1)).map(|error| error.root()),
            Some("the launch failed")
        );
    }

    /// The read is where the failure is delivered, and it must name the root
    /// cause rather than a bare missing handle.
    #[test]
    fn reading_a_claimed_tensor_reports_the_root_cause() {
        let mut container = HandleContainer::<String>::new();
        container.claim(tid(1), TensorError::new("the kernel failed to compile"));

        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            container.get_handle(&tid(1), &TensorStatus::ReadWrite)
        }));
        let payload = read.expect_err("a claimed tensor must not read back");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or_default();
        assert!(
            message.contains("never written") && message.contains("failed to compile"),
            "got: {message}"
        );

        // The claim survives the read that tripped over it: an unwind past a
        // `remove_entry` would leave the next read reporting a missing handle
        // instead of the reason.
        assert!(container.error(&tid(1)).is_some());
    }

    /// Propagation shares the root, so a read below a chain of skipped work
    /// still names the failure that started it.
    #[test]
    fn propagation_keeps_the_root_and_counts_the_hops() {
        let root = TensorError::new("the kernel failed to compile");
        let one = root.propagated();
        let two = one.propagated();

        assert!(root.same_failure(&two));
        assert_eq!((root.depth(), one.depth(), two.depth()), (0, 1, 2));
        assert_eq!(two.root(), "the kernel failed to compile");
        assert!(
            !root.same_failure(&TensorError::new("the kernel failed to compile")),
            "same message, different failure"
        );
    }

    /// The claim is released by the tensor's own `Drop`, like any other
    /// handle — which is what bounds the set to the tensors still alive.
    #[test]
    fn a_claim_is_released_with_its_tensor() {
        let mut container = HandleContainer::<String>::new();
        container.claim(tid(1), TensorError::new("boom"));

        container.free(&TensorIr {
            id: tid(1),
            shape: burn_backend::Shape::new([1]),
            status: TensorStatus::ReadWrite,
            dtype: burn_backend::DType::F32,
        });

        assert!(container.error(&tid(1)).is_none());
        assert_eq!(container.num_handles(), 0);
    }

    /// `has_claims` gates every claim check, so drift disables the whole
    /// mechanism silently — the skip stops happening and failed work starts
    /// computing on bytes nothing wrote. Every path that can change an entry
    /// is exercised here against the map it is supposed to mirror.
    #[test]
    fn the_claim_count_cannot_drift_from_the_map() {
        fn claimed<H: Clone>(container: &HandleContainer<H>) -> usize {
            container
                .handles
                .values()
                .filter(|handle| matches!(handle, Handle::Errored(_)))
                .count()
        }
        fn check<H: Clone>(container: &HandleContainer<H>, at: &str) {
            let actual = claimed(container);
            assert_eq!(container.claimed, actual, "count drifted after {at}");
            assert_eq!(
                container.has_claims(),
                actual > 0,
                "has_claims wrong at {at}"
            );
        }

        let mut container = HandleContainer::<String>::new();
        let error = || TensorError::new("boom");
        check(&container, "empty");

        // Claim, then claim the same id again: one entry, one claim.
        container.claim(tid(1), error());
        check(&container, "claim");
        container.claim(tid(1), error());
        check(&container, "claim again");

        // Data displacing a claim, and a claim displacing data.
        container.register_handle(tid(1), "written".to_string());
        check(&container, "register over a claim");
        container.claim(tid(1), error());
        check(&container, "claim over data");

        // The conservative claim, both ways.
        container.register_handle(tid(2), "written".to_string());
        container.claim_unwritten(tid(2), error());
        check(&container, "claim_unwritten over data");
        container.claim_unwritten(tid(3), error());
        check(&container, "claim_unwritten over nothing");

        // Both removal paths.
        container.remove_handle(tid(3));
        check(&container, "remove_handle");
        container.free(&TensorIr {
            id: tid(1),
            shape: burn_backend::Shape::new([1]),
            status: TensorStatus::ReadWrite,
            dtype: burn_backend::DType::F32,
        });
        check(&container, "free");

        // A read that keeps its entry, and one that consumes it.
        container.register_handle(tid(4), "written".to_string());
        container.get_handle(&tid(4), &TensorStatus::ReadOnly);
        check(&container, "get_handle ReadOnly");
        container.get_handle(&tid(4), &TensorStatus::ReadWrite);
        check(&container, "get_handle ReadWrite");

        // A fork carries the count it copied.
        container.claim(tid(5), error());
        let fork = container.fork();
        check(&fork, "fork");
        assert!(fork.has_claims());
    }

    #[test]
    fn double_fork_is_fully_isolated() {
        // Simulates what happens when UnsafeTuneContext::get() is called on a Fork:
        // it forks again, creating a second level of isolation.
        let mut container = HandleContainer::<String>::new();
        container.register_handle(tid(1), "input".to_string());

        let fork1 = container.fork();
        let mut fork2 = fork1.fork();

        fork2.register_handle(tid(200), "deep_output".to_string());

        assert!(fork1.get_handle_ref(&tid(200)).is_none());
        assert!(container.get_handle_ref(&tid(200)).is_none());
        assert!(fork2.get_handle_ref(&tid(200)).is_some());
    }
}
