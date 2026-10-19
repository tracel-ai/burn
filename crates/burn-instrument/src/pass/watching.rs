//! Installing a pass's recorder as the runtimes' observer, one pass at a time.
//!
//! The runtime's slot is process-wide, and it has to be: a launch does not
//! arrive on the thread that entered the components — burn's fusion processes a
//! stream off the thread that recorded it — so an observer scoped to one thread
//! would see nothing at all. The cost is that a second pass watched at once
//! would write into the first's tree, so a second one is refused.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use burn_fusion::observer::FusionObservation;
use cubecl::logging::LaunchObservation;

use super::recorder::PassRecorder;

/// What watching a pass installed: the runtime's launch observer and the
/// fusion server's. Both come out when it drops.
pub struct Watching {
    _launches: LaunchObservation,
    _fusion: FusionObservation,
    /// Last, since fields drop in order: the claim is given up only once both
    /// observers above have been put back, so the next pass to claim it
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
    pub fn new(recorder: &Arc<PassRecorder>, claim: Claim) -> Self {
        Self {
            _launches: LaunchObservation::new(recorder.clone()),
            _fusion: FusionObservation::new(recorder.clone()),
            _claim: claim,
        }
    }
}

/// Whether a pass is being watched anywhere in the process.
static WATCHING: AtomicBool = AtomicBool::new(false);

/// This process's one collected pass. Given up on drop.
pub struct Claim;

impl Claim {
    /// # Panics
    ///
    /// If another pass holds it: the observer slots are process-wide, so a
    /// second collection would record the first's kernels and the first would
    /// lose its own.
    pub fn take() -> Self {
        let already = WATCHING.swap(true, Ordering::AcqRel);
        assert!(
            !already,
            "a pass is already being collected in this process; the runtime's launch observer is \
             process-wide, so collections run one at a time"
        );
        Claim
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        WATCHING.store(false, Ordering::Release);
    }
}
