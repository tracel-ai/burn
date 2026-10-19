use std::sync::Arc;

use burn_tensor::Device;

use super::recorder::PassRecorder;
use super::watching::{Claim, Watching};
use crate::{Breakdown, ComponentObservation};

/// Collects one pass on a device into a [`Breakdown`].
///
/// See the [crate docs](crate#collecting-a-pass) for what a collected pass
/// records and how far its numbers are to be trusted.
pub struct PassCollector {
    recorder: PassRecorder,
    claim: Claim,
}

impl PassCollector {
    /// A collector for work running on `device`, and the process's one until
    /// it is dropped.
    ///
    /// # Panics
    ///
    /// If another collector exists in the process. Refused here, before the
    /// device is synced or its fusion worker pointed anywhere: either would
    /// disturb the pass the other is collecting.
    ///
    /// If `device` fuses and this crate was built without its `fusion`
    /// feature: the collector could not see the blocks, and would place every
    /// fused kernel on whichever component sealed it.
    pub fn new(device: &Device) -> Self {
        #[cfg(not(feature = "fusion"))]
        assert!(
            device.as_dispatch().fusion_worker().is_none(),
            "the device fuses, so collecting it needs burn-instrument's `fusion` feature"
        );
        let claim = Claim::take();
        Self {
            recorder: PassRecorder::new(device),
            claim,
        }
    }

    /// Run `pass` collected, and end it: every component it enters arrives
    /// from this thread and every kernel the runtime issues for it from
    /// whichever thread issued it, until the pass's last queued segment has
    /// run and its measurements are read back.
    ///
    /// Collected, the pass runs the graph it runs in production — nothing is
    /// drained at a boundary — so call it on work already synced: an operation
    /// queued before it would be watched too.
    ///
    /// Taken by value: a collector holds one pass, and its breakdown is read
    /// once that pass has ended.
    ///
    /// # Panics
    ///
    /// If the device stops answering at the end of the pass, or if the work
    /// producing an observed value failed.
    pub fn collect(self, pass: impl FnOnce()) -> Breakdown {
        // One recorder, three observations: the components arrive on this
        // thread and the kernels and fusion blocks on whichever thread runs
        // them, so the tree is written from both. Each kernel lands under the
        // attribution of the thread that issued it — see `LaunchCursor`.
        let recorder = Arc::new(self.recorder);
        let watching = Watching::new(&recorder, self.claim);
        let observation = ComponentObservation::new(Box::new(Arc::clone(&recorder)));
        pass();
        drop(observation);
        recorder.finish(watching)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A second collector is refused before it touches the device: built while
    /// another exists, it would sync the device and point the fusion worker
    /// away from the pass the first is collecting, before `collect` could
    /// refuse it.
    #[test]
    fn a_second_collector_is_refused_before_it_touches_the_device() {
        let device = Device::flex();
        let first = PassCollector::new(&device);
        let second = std::panic::catch_unwind(|| PassCollector::new(&Device::flex()));
        assert!(second.is_err(), "refused while the first exists");

        drop(first);
        drop(PassCollector::new(&device));
    }
}
