use crate::stream::Context;
use burn_backend::ExecutionError;
use burn_ir::{HandleContainer, OperationIr, TensorError};

/// The message inside a caught panic payload. Covers what `panic!` produces:
/// `&'static str` and `String`.
pub fn panic_message(panic: &(dyn core::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>")
}

/// The payload of a panic the scope caught, kept only so a caller can log it.
/// Every failure's real report is the claim it left on the tensors.
pub type Panic = Box<dyn core::any::Any + Send>;

/// What a scope's work did.
pub enum Outcome {
    /// It ran and wrote its outputs.
    Ran,
    /// It did not run, because an input it needed carried a failure. Its write
    /// set carries that same failure now, so a read of one of those tensors
    /// names the cause that started it rather than one of its own.
    Skipped,
    /// It ran and reported a failure. Its write set carries the error it
    /// reported, whole.
    Reported,
    /// It ran and panicked. Its write set carries the panic's message, and the
    /// payload comes back so a caller can log it.
    Panicked(Panic),
}

/// Whether a panic out of the body is this scope's to catch.
pub enum OnPanic {
    /// Catch it. The payload comes back as [`Outcome::Panicked`], and the
    /// write set is claimed before it does — which is why the catch lives in
    /// the scope rather than at the call site.
    Catch,
    /// Let it out, for an outer scope to see. For work that already runs
    /// inside one: a fallback in the middle of a fused block is one operation
    /// of a kernel that is still part way through, so swallowing its panic
    /// would let the block carry on as though the piece it could not serve had
    /// run. The claim still happens, on the way out, through [`Drop`].
    ///
    /// A reported failure is claimed here all the same — only a panic is left
    /// to the outer scope, because only a panic is what the outer unit cannot
    /// carry on through.
    Raise,
}

/// Whatever a scope can reach the handle container through.
///
/// The unfused path runs against the container itself; a fused block runs
/// against the whole [`Context`], because that is what the optimization needs.
/// The scope only ever wants the handles, so it asks for them rather than
/// being written twice.
pub trait Handles {
    /// What the container holds.
    type Handle: Clone;

    /// The container the claim is recorded in.
    fn handles(&mut self) -> &mut HandleContainer<Self::Handle>;
}

impl<H: Clone> Handles for HandleContainer<H> {
    type Handle = H;

    fn handles(&mut self) -> &mut HandleContainer<H> {
        self
    }
}

impl<H: Clone> Handles for Context<H> {
    type Handle = H;

    fn handles(&mut self) -> &mut HandleContainer<H> {
        &mut self.handles
    }
}

/// What one scope covers: the operations it reads and the tensors it writes.
enum Work<'a> {
    /// One operation.
    One(&'a OperationIr),
    /// A fused block, as an index into the segment's IR.
    ///
    /// A fused kernel is one unit of work: it reads every input of every
    /// operation it replaced and writes every output. So one claimed input
    /// anywhere in it stops the whole thing, and a failure anywhere in it
    /// leaves the whole write set unwritten.
    Block {
        ir: &'a [OperationIr],
        ordering: &'a [usize],
    },
}

impl Work<'_> {
    fn operations(&self) -> impl Iterator<Item = &OperationIr> {
        let (one, block) = match self {
            Work::One(ir) => (Some(*ir), None),
            Work::Block { ir, ordering } => (None, Some(ordering.iter().map(move |id| &ir[*id]))),
        };

        one.into_iter().chain(block.into_iter().flatten())
    }

    fn input_error<H: Clone>(&self, handles: &HandleContainer<H>) -> Option<TensorError> {
        self.operations()
            .find_map(|op| input_error(op, handles))
            .map(TensorError::propagated)
    }

    fn claim<H: Clone>(&self, handles: &mut HandleContainer<H>, error: &TensorError) {
        for op in self.operations() {
            set_output_errors(op, handles, error);
        }
    }
}

/// Where a scope is between its one entry and its one exit.
enum State {
    /// Opened on a claimed input. The write set took that failure in the
    /// constructor, and the body does not run.
    Skipped,
    /// Entered, and not yet past the exit. A scope dropped in this state was
    /// left without reaching one, and claims its write set on the way out.
    Running,
    /// Past the exit, so whatever was going to be claimed has been.
    Finished,
}

/// One unit of work, and the claim on everything it was going to write.
pub struct WriteScope<'a, W: Handles> {
    work: Work<'a>,
    target: &'a mut W,
    state: State,
}

impl<'a, W: Handles> WriteScope<'a, W> {
    /// Open a scope over one operation.
    pub fn over(ir: &'a OperationIr, target: &'a mut W) -> Self {
        Self::open(Work::One(ir), target)
    }

    /// Open a scope over a fused block, which is one unit of work covering
    /// every operation at `ordering`.
    pub fn over_block(ir: &'a [OperationIr], ordering: &'a [usize], target: &'a mut W) -> Self {
        Self::open(Work::Block { ir, ordering }, target)
    }

    /// Decide, once, whether this is a skip or an entry. The write set is the
    /// work's own outputs, so it is never approximate — which is what lets
    /// every claim displace, and why no caller has to say what to do about a
    /// handle that is already registered.
    fn open(work: Work<'a>, target: &'a mut W) -> Self {
        let state = match work.input_error(target.handles()) {
            Some(error) => {
                work.claim(target.handles(), &error);
                State::Skipped
            }
            None => State::Running,
        };

        Self {
            work,
            target,
            state,
        }
    }

    /// Run `body` between the entry and the exit, and claim the write set if
    /// it does not reach the end.
    ///
    /// The catch lives here rather than at the call site: a unit of work that
    /// fails has to claim its write set, and putting the catch where the write
    /// set is known is what stops that being four separate things a caller can
    /// get wrong. `on_panic` says which scope owns the catch — this one, or
    /// the one this work is already running inside.
    pub fn run(
        mut self,
        on_panic: OnPanic,
        body: impl FnOnce(&mut W) -> Result<(), ExecutionError>,
    ) -> Outcome {
        // `Drop` claims only what is still `Running`, so a skip needs nothing
        // more than to not run.
        if let State::Skipped = self.state {
            return Outcome::Skipped;
        }

        let target = &mut *self.target;
        let ran = match on_panic {
            OnPanic::Catch => {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(target)))
            }
            // A panic unwinds out of here, leaving the state `Running`, so the
            // claim is made by `Drop` on the way past.
            OnPanic::Raise => Ok(body(target)),
        };
        self.state = State::Finished;

        match ran {
            Ok(Ok(())) => Outcome::Ran,
            Ok(Err(error)) => {
                let error = TensorError::new(error);
                self.work.claim(self.target.handles(), &error);
                Outcome::Reported
            }
            Err(panic) => {
                let error =
                    TensorError::new(ExecutionError::generic(panic_message(panic.as_ref())));
                self.work.claim(self.target.handles(), &error);
                Outcome::Panicked(panic)
            }
        }
    }
}

impl<W: Handles> Drop for WriteScope<'_, W> {
    fn drop(&mut self) {
        if !matches!(self.state, State::Running) {
            return;
        }

        // Nothing reached an exit — a panic unwinding through work whose scope
        // does not catch, or raised before the body. There is no payload to
        // name here, but the claim still has to be made: a read of one of these
        // tensors must fail rather than hand back whatever the allocation
        // happened to hold.
        let error = TensorError::new(ExecutionError::with_context(
            "the work that was going to write it did not reach the end",
        ));
        let work = &self.work;
        work.claim(self.target.handles(), &error);
    }
}

/// The failure that errored any tensor `op` reads — the check a unit of work
/// makes before it runs.
///
/// Work whose input was never written must not run: those bytes are whatever
/// the allocation happened to hold, and computing on them turns a failure
/// that named one tensor into a wrong answer that names none. The outputs
/// take the same error instead, so a read below the skip still reports the
/// failure that started it.
///
/// `inputs()` rather than `nodes()`: this runs before every operation on the
/// hot path, and `nodes()` collects into a fresh `Vec` to chain the two.
fn input_error<'a, H>(op: &OperationIr, handles: &'a HandleContainer<H>) -> Option<&'a TensorError>
where
    H: Clone,
{
    // No tensor is errored, so nothing can be found — and asking anyway would
    // cost a boxed iterator per operation for an answer that is always
    // `None`. This runs before every operation, so the check has to be free
    // while nothing has failed.
    if !handles.has_errors() {
        return None;
    }

    // A drop names its tensor as an input, but it does not read it — it is
    // what releases it, and releasing is how an error stops being held. Skip
    // it and the error outlives every tensor that could report it, for the
    // life of the server: the bound this whole design rests on is that an
    // error lives exactly as long as the tensor carrying it.
    if let OperationIr::Drop(_) = op {
        return None;
    }

    op.inputs().find_map(|node| handles.error(&node.id))
}

/// Record `error` on every tensor `op` was going to write, so a read of one
/// reports it instead of handing back bytes nothing wrote.
fn set_output_errors<H>(op: &OperationIr, handles: &mut HandleContainer<H>, error: &TensorError)
where
    H: Clone,
{
    for node in op.outputs() {
        handles.set_error(node.id, error.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{DType, Shape};
    use burn_ir::{FloatOperationIr, TensorId, TensorIr, TensorStatus, UnaryOpIr};

    fn tensor(id: u64, status: TensorStatus) -> TensorIr {
        TensorIr {
            id: TensorId::new(id),
            status,
            shape: Shape::from(vec![1]),
            dtype: DType::F32,
        }
    }

    /// One operation reading `input` and writing `out`.
    fn exp(input: u64, out: u64) -> OperationIr {
        OperationIr::Float(
            DType::F32,
            FloatOperationIr::Exp(UnaryOpIr {
                input: tensor(input, TensorStatus::ReadOnly),
                out: tensor(out, TensorStatus::NotInit),
            }),
        )
    }

    fn container() -> HandleContainer<String> {
        let mut handles = HandleContainer::new();
        handles.register_handle(TensorId::new(0), "input".to_string());
        handles
    }

    /// The success path leaves the write set exactly as the work left it.
    #[test]
    fn work_that_runs_claims_nothing() {
        let ir = exp(0, 1);
        let mut handles = container();

        let outcome = WriteScope::over(&ir, &mut handles).run(OnPanic::Catch, |handles| {
            handles.register_handle(TensorId::new(1), "written".to_string());
            Ok(())
        });

        assert!(matches!(outcome, Outcome::Ran));
        assert!(
            !handles.has_errors(),
            "nothing failed, so nothing is claimed"
        );
        assert!(handles.has_handle(&TensorId::new(1)));
    }

    /// Work that reports a failure claims its write set, and the claim carries
    /// the error it reported — the path no call site can reach until
    /// `Operation::execute` is fallible.
    #[test]
    fn work_that_reports_a_failure_claims_its_write_set() {
        let ir = exp(0, 1);
        let mut handles = container();

        let outcome = WriteScope::over(&ir, &mut handles).run(OnPanic::Catch, |_handles| {
            Err(ExecutionError::generic("the kernel failed to compile"))
        });

        assert!(matches!(outcome, Outcome::Reported), "reported, not raised");
        let claim = handles
            .error(&TensorId::new(1))
            .expect("its output is claimed");
        assert_eq!(claim.root(), "the kernel failed to compile");
        assert_eq!(claim.depth(), 0, "this is where the failure happened");
        assert!(
            matches!(claim.cause(), ExecutionError::Generic { .. }),
            "the reported error is carried whole"
        );
    }

    /// Work that panics claims the same way, and hands the payload back so the
    /// caller can log it. The claim, not the payload, is the report.
    #[test]
    fn work_that_panics_claims_its_write_set() {
        let ir = exp(0, 1);
        let mut handles = container();

        let outcome = WriteScope::over(&ir, &mut handles)
            .run(OnPanic::Catch, |_handles| panic!("this kernel cannot run"));

        assert!(
            matches!(outcome, Outcome::Panicked(_)),
            "raised, not reported"
        );
        assert_eq!(
            handles
                .error(&TensorId::new(1))
                .expect("its output is claimed")
                .root(),
            "this kernel cannot run"
        );
    }

    /// A claimed input opens the scope on a skip: the body never runs, and the
    /// write set takes that same failure one hop down.
    #[test]
    fn a_claimed_input_skips_the_body() {
        let ir = exp(0, 1);
        let mut handles = container();
        let root = TensorError::new(ExecutionError::generic("an earlier kernel failed"));
        handles.set_error(TensorId::new(0), root.clone());

        let outcome = WriteScope::over(&ir, &mut handles)
            .run(OnPanic::Catch, |_handles| panic!("the body must not run"));

        assert!(matches!(outcome, Outcome::Skipped));
        let claim = handles
            .error(&TensorId::new(1))
            .expect("its output is claimed");
        assert!(
            claim.same_root(&root),
            "it names the failure that started it"
        );
        assert_eq!(claim.depth(), 1, "one hop below that failure");
    }

    /// A block is one unit of work: one claimed input anywhere in it stops the
    /// whole thing, and every operation's outputs take the failure together.
    #[test]
    fn one_claimed_input_stops_a_whole_block() {
        let ir = vec![exp(0, 1), exp(9, 2)];
        let mut handles = container();
        handles.register_handle(TensorId::new(9), "second input".to_string());
        let root = TensorError::new(ExecutionError::generic("an earlier kernel failed"));
        // Claimed input of the *second* operation only.
        handles.set_error(TensorId::new(9), root.clone());

        let outcome = WriteScope::over_block(&ir, &[0, 1], &mut handles)
            .run(OnPanic::Catch, |_handles| panic!("the block must not run"));

        assert!(matches!(outcome, Outcome::Skipped));
        for out in [1, 2] {
            let claim = handles
                .error(&TensorId::new(out))
                .expect("every output of the block is claimed");
            assert!(claim.same_root(&root));
        }
    }

    /// A reported failure is claimed under [`OnPanic::Raise`] too — only a
    /// panic is left to the outer scope.
    #[test]
    fn raising_work_claims_a_reported_failure() {
        let ir = exp(0, 1);
        let mut handles = container();

        let outcome = WriteScope::over(&ir, &mut handles).run(OnPanic::Raise, |_handles| {
            Err(ExecutionError::generic("it declined to run"))
        });

        assert!(matches!(outcome, Outcome::Reported));
        assert_eq!(
            handles
                .error(&TensorId::new(1))
                .expect("its output is claimed")
                .root(),
            "it declined to run"
        );
    }

    /// [`OnPanic::Raise`] lets the panic out for an outer scope to see, and
    /// still claims on the way through.
    #[test]
    fn raising_work_claims_on_its_way_out() {
        let ir = exp(0, 1);
        let mut handles = container();

        let escaped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WriteScope::over(&ir, &mut handles)
                .run(OnPanic::Raise, |_handles| panic!("cannot serve it"));
        }));

        assert!(escaped.is_err(), "the panic reaches the caller");
        assert!(
            handles.error(&TensorId::new(1)).is_some(),
            "and the write set is claimed regardless"
        );
    }
}
