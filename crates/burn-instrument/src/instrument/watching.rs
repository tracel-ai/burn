//! Installing a run's recorder as the runtimes' observer, one run at a time.
//!
//! The runtime's slot is process-wide, and it has to be: a launch does not
//! arrive on the thread that entered the components — burn's fusion processes a
//! stream off the thread that recorded it — so an observer scoped to one thread
//! would see nothing at all. The cost is that a second run watched at once
//! would write into the first's tree, so a second one is refused.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "fusion")]
use burn_fusion::observer::FusionObservation;
use cubecl::logging::LaunchObservation;

use super::recorder::RunRecorder;

/// What watching a run installed: the runtime's launch observer and the
/// fusion server's. Both come out when it drops.
pub struct Watching {
    _launches: LaunchObservation,
    #[cfg(feature = "fusion")]
    _fusion: FusionObservation,
    /// Last, since fields drop in order: the claim is given up only once both
    /// observers above have been put back, so the next run to claim it
    /// cannot have its own removed by this one's.
    _claim: Claim,
}

impl Watching {
    /// Hand every launch, and every operation and block the fusion server
    /// runs, to `recorder` for as long as the returned guard lives.
    ///
    /// The guards are the runtimes' own: each holds one process-wide slot,
    /// replaces whatever was installed there for its lifetime, and puts it
    /// back on drop.
    ///
    /// `claim` is held until both observers have been put back.
    pub fn new(recorder: &Arc<RunRecorder>, claim: Claim) -> Self {
        Self {
            _launches: LaunchObservation::new(recorder.clone()),
            #[cfg(feature = "fusion")]
            _fusion: FusionObservation::new(recorder.clone()),
            _claim: claim,
        }
    }
}

/// Whether a run is being watched anywhere in the process.
static WATCHING: AtomicBool = AtomicBool::new(false);

/// This process's one instrumented run. Given up on drop.
pub struct Claim;

impl Claim {
    /// # Panics
    ///
    /// If another run holds it: the observer slots are process-wide, so a
    /// second instrument would record the first's kernels and the first would
    /// lose its own.
    pub fn take() -> Self {
        let already = WATCHING.swap(true, Ordering::AcqRel);
        assert!(
            !already,
            "a run is already being instrumented in this process; the runtime's launch observer \
             is process-wide, so instruments run one at a time"
        );
        Claim
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        WATCHING.store(false, Ordering::Release);
    }
}
