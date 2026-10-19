use std::sync::Arc;

use burn_tensor::Device;

use super::recorder::PassRecorder;
use super::watching::Watching;
use crate::Breakdown;

/// Collects one pass on a device into a [`Breakdown`].
///
/// See the [crate docs](crate#collecting-a-pass) for what a collected pass
/// records and how far its numbers are to be trusted.
pub struct PassCollector(PassRecorder);

impl PassCollector {
    /// A collector for work running on `device`.
    pub fn new(device: &Device) -> Self {
        Self(PassRecorder::new(device))
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
    /// If another pass is being collected in the process (the runtimes'
    /// observer slots are process-wide), if the device stops answering at the
    /// end of the pass, or if the work producing an observed value failed.
    pub fn collect(self, pass: impl FnOnce()) -> Breakdown {
        // Installed three times over: the components arrive on this thread
        // and the kernels and fusion blocks on whichever thread runs them, so
        // the tree is written from both. Each kernel lands under the
        // attribution of the thread that issued it — see `LaunchCursor`.
        let collector = Arc::new(self.0);
        let watching = Watching::new(&collector);
        let collecting = crate::collect(Box::new(Arc::clone(&collector)));
        pass();
        drop(collecting);
        // While the kernels are still being watched: the pass's last fused
        // segment is still queued, and it runs at the next sync. Every
        // measurement the pass kept is read back there too.
        collector.finish();
        drop(watching);
        // Unwatched, since reading the values back launches kernels too.
        collector.price_observed();
        collector.breakdown()
    }
}
