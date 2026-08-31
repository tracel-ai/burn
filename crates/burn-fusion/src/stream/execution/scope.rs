//! One scope around one unit of work.
//!
//! Without it, every execution site keeps the same bookkeeping by hand: decide
//! whether the inputs can be trusted, claim the write set on each failure path,
//! leave it alone on the success path, catch the panic, and hope no early
//! return forgot one of the four. The scope makes forgetting loud instead of
//! silent — there is one way in, one way out, and the claim is not something a
//! caller can omit.
//!
//! A scope is opened one of two ways, and the choice is made once, in the
//! constructor, which is why the two can never interleave:
//!
//! - work whose inputs all read cleanly **enters**, and runs;
//! - work whose input a failure claims **skips**: its write set takes that same
//!   failure one hop down, and the body never runs.
//!
//! The body is a closure rather than the scope being a guard value, because a
//! guard enforces nothing here: `#[must_use]` says nothing about a bound value
//! on a path that returns early. A closure has exactly one exit, and the scope
//! owns what happens at it.

use super::{input_error, panic_message, set_output_errors};
use crate::stream::Context;
use burn_backend::ExecutionError;
use burn_ir::{HandleContainer, OperationIr, TensorError};

/// The payload of a panic the scope caught, kept only so a caller can log it.
/// Every failure's real report is the claim it left on the tensors.
pub(crate) type Panic = Box<dyn core::any::Any + Send>;

/// What a scope's work did.
pub(crate) enum Outcome<T> {
    /// It ran and wrote its outputs.
    Ran(T),
    /// It did not run, because an input it needed carried a failure. Its write
    /// set carries that same failure now, so a read of one of those tensors
    /// names the cause that started it rather than one of its own.
    Skipped,
    /// It ran and failed. Its write set carries the failure. The payload is
    /// `Some` when the work panicked rather than reporting.
    Failed(Option<Panic>),
}

/// Whatever a scope can reach the handle container through.
///
/// The unfused path runs against the container itself; a fused block runs
/// against the whole [`Context`], because that is what the optimization needs.
/// The scope only ever wants the handles, so it asks for them rather than
/// being written twice.
pub(crate) trait Claims<H: Clone> {
    /// The container the claim is recorded in.
    fn handles(&mut self) -> &mut HandleContainer<H>;
}

impl<H: Clone> Claims<H> for HandleContainer<H> {
    fn handles(&mut self) -> &mut HandleContainer<H> {
        self
    }
}

impl<H: Clone> Claims<H> for Context<H> {
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

/// One unit of work, and the claim on everything it was going to write.
pub(crate) struct WriteScope<'a, H: Clone, W: Claims<H>> {
    work: Work<'a>,
    target: &'a mut W,
    /// Set between entry and exit. A scope dropped while still armed was left
    /// without reaching either exit, and claims its write set on the way out.
    armed: bool,
    /// Whether the body may run at all. False for a scope opened on a skip.
    entered: bool,
    _handle: core::marker::PhantomData<H>,
}

impl<'a, H: Clone, W: Claims<H>> WriteScope<'a, H, W> {
    /// Open a scope over one operation.
    pub(crate) fn over(ir: &'a OperationIr, target: &'a mut W) -> Self {
        Self::open(Work::One(ir), target)
    }

    /// Open a scope over a fused block, which is one unit of work covering
    /// every operation at `ordering`.
    pub(crate) fn over_block(
        ir: &'a [OperationIr],
        ordering: &'a [usize],
        target: &'a mut W,
    ) -> Self {
        Self::open(Work::Block { ir, ordering }, target)
    }

    /// Decide, once, whether this is a skip or an entry. The write set is the
    /// work's own outputs, so it is never approximate — which is what lets
    /// every claim displace, and why no caller has to say what to do about a
    /// handle that is already registered.
    fn open(work: Work<'a>, target: &'a mut W) -> Self {
        match work.input_error(target.handles()) {
            Some(error) => {
                work.claim(target.handles(), &error);
                Self {
                    work,
                    target,
                    armed: false,
                    entered: false,
                    _handle: core::marker::PhantomData,
                }
            }
            None => Self {
                work,
                target,
                armed: true,
                entered: true,
                _handle: core::marker::PhantomData,
            },
        }
    }

    /// Run `body` between the entry and the exit.
    ///
    /// The catch lives here rather than at the call site: a unit of work that
    /// panics has to claim its write set, and putting the catch where the write
    /// set is known is what stops that being four separate things a caller can
    /// get wrong.
    pub(crate) fn run<T>(
        mut self,
        body: impl FnOnce(&mut W) -> Result<T, ExecutionError>,
    ) -> Outcome<T> {
        if !self.entered {
            self.armed = false;
            return Outcome::Skipped;
        }

        let target = &mut *self.target;
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(target)));
        self.armed = false;

        match ran {
            Ok(Ok(value)) => Outcome::Ran(value),
            Ok(Err(error)) => {
                let error = TensorError::new(error);
                self.work.claim(self.target.handles(), &error);
                Outcome::Failed(None)
            }
            Err(panic) => {
                let error = TensorError::panicked(panic_message(panic.as_ref()));
                self.work.claim(self.target.handles(), &error);
                Outcome::Failed(Some(panic))
            }
        }
    }

    /// Run `body`, letting a panic out rather than catching it.
    ///
    /// For work that already runs inside another scope: a fallback in the
    /// middle of a fused block is one operation of a kernel that is still
    /// part way through, so swallowing its panic would let the block carry on
    /// as though the piece it could not serve had run. The claim still
    /// happens — on the way out, through [`Drop`].
    ///
    /// `None` when an input was claimed and the body never ran.
    pub(crate) fn run_raising<T>(mut self, body: impl FnOnce(&mut W) -> T) -> Option<T> {
        if !self.entered {
            self.armed = false;
            return None;
        }

        let value = body(self.target);
        self.armed = false;

        Some(value)
    }
}

impl<H: Clone, W: Claims<H>> Drop for WriteScope<'_, H, W> {
    fn drop(&mut self) {
        if !self.armed {
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

        let outcome = WriteScope::over(&ir, &mut handles).run(|handles| {
            handles.register_handle(TensorId::new(1), "written".to_string());
            Ok(())
        });

        assert!(matches!(outcome, Outcome::Ran(())));
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

        let outcome = WriteScope::over(&ir, &mut handles)
            .run(|_handles| Err::<(), _>(ExecutionError::generic("the kernel failed to compile")));

        assert!(
            matches!(outcome, Outcome::Failed(None)),
            "reported, not raised"
        );
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
            .run(|_handles| -> Result<(), ExecutionError> { panic!("this kernel cannot run") });

        assert!(
            matches!(outcome, Outcome::Failed(Some(_))),
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
            .run(|_handles| -> Result<(), ExecutionError> { panic!("the body must not run") });

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
            .run(|_handles| -> Result<(), ExecutionError> { panic!("the block must not run") });

        assert!(matches!(outcome, Outcome::Skipped));
        for out in [1, 2] {
            let claim = handles
                .error(&TensorId::new(out))
                .expect("every output of the block is claimed");
            assert!(claim.same_root(&root));
        }
    }

    /// `run_raising` lets the panic out for an outer scope to see, and still
    /// claims on the way through.
    #[test]
    fn raising_work_claims_on_its_way_out() {
        let ir = exp(0, 1);
        let mut handles = container();

        let escaped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WriteScope::over(&ir, &mut handles).run_raising(|_handles| panic!("cannot serve it"));
        }));

        assert!(escaped.is_err(), "the panic reaches the caller");
        assert!(
            handles.error(&TensorId::new(1)).is_some(),
            "and the write set is claimed regardless"
        );
    }
}
