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

/// One unit of work, and the claim on everything it was going to write.
pub(crate) struct WriteScope<'a, H: Clone> {
    ir: &'a OperationIr,
    handles: &'a mut HandleContainer<H>,
    /// Set between entry and exit. A scope dropped while still armed was left
    /// without reaching either exit, and claims its write set on the way out.
    armed: bool,
    /// Whether the body may run at all. False for a scope opened on a skip.
    entered: bool,
}

impl<'a, H: Clone> WriteScope<'a, H> {
    /// Open a scope over `ir`, deciding once whether this is a skip or an
    /// entry. The write set is `ir`'s outputs, so it is never approximate —
    /// which is what lets every claim displace, and why no caller has to say
    /// what to do about a handle that is already there.
    pub(crate) fn over(ir: &'a OperationIr, handles: &'a mut HandleContainer<H>) -> Self {
        let skip = input_error(ir, handles).map(TensorError::propagated);

        match skip {
            Some(error) => {
                set_output_errors(ir, handles, &error);
                Self {
                    ir,
                    handles,
                    armed: false,
                    entered: false,
                }
            }
            None => Self {
                ir,
                handles,
                armed: true,
                entered: true,
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
        body: impl FnOnce(&mut HandleContainer<H>) -> Result<T, ExecutionError>,
    ) -> Outcome<T> {
        if !self.entered {
            self.armed = false;
            return Outcome::Skipped;
        }

        let handles = &mut *self.handles;
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(handles)));
        self.armed = false;

        match ran {
            Ok(Ok(value)) => Outcome::Ran(value),
            Ok(Err(error)) => {
                set_output_errors(self.ir, self.handles, &TensorError::new(error));
                Outcome::Failed(None)
            }
            Err(panic) => {
                let error = TensorError::panicked(panic_message(panic.as_ref()));
                set_output_errors(self.ir, self.handles, &error);
                Outcome::Failed(Some(panic))
            }
        }
    }
}

impl<H: Clone> Drop for WriteScope<'_, H> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }

        // Nothing reached an exit — a panic raised between the entry and the
        // body, where there is no payload to name. The claim still has to be
        // made: a read of one of these tensors must fail rather than hand back
        // whatever the allocation happened to hold.
        let error = TensorError::new(ExecutionError::with_context(
            "the work that was going to write it did not reach the end",
        ));
        set_output_errors(self.ir, self.handles, &error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{DType, Shape};
    use burn_ir::{ExistingHandle, FloatOperationIr, TensorId, TensorIr, TensorStatus, UnaryOpIr};

    fn tensor(id: u64, status: TensorStatus) -> TensorIr {
        TensorIr {
            id: TensorId::new(id),
            status,
            shape: Shape::from(vec![1]),
            dtype: DType::F32,
        }
    }

    /// One operation reading tensor 0 and writing tensor 1.
    fn exp() -> OperationIr {
        OperationIr::Float(
            DType::F32,
            FloatOperationIr::Exp(UnaryOpIr {
                input: tensor(0, TensorStatus::ReadOnly),
                out: tensor(1, TensorStatus::NotInit),
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
        let ir = exp();
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
        let ir = exp();
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
        let ir = exp();
        let mut handles = container();

        let outcome = WriteScope::over(&ir, &mut handles)
            .run(|_handles| -> Result<(), ExecutionError> { panic!("this kernel cannot run") });

        assert!(
            matches!(outcome, Outcome::Failed(Some(_))),
            "raised, not reported"
        );
        let claim = handles
            .error(&TensorId::new(1))
            .expect("its output is claimed");
        assert_eq!(claim.root(), "this kernel cannot run");
    }

    /// A claimed input opens the scope on a skip: the body never runs, and the
    /// write set takes that same failure one hop down.
    #[test]
    fn a_claimed_input_skips_the_body() {
        let ir = exp();
        let mut handles = container();
        let root = TensorError::new(ExecutionError::generic("an earlier kernel failed"));
        handles.set_error(TensorId::new(0), root.clone(), ExistingHandle::Displace);

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
}
