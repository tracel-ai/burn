use burn_cubecl::cubecl::logging::{LaunchObservation, LaunchObserver};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// Observe actual AdamW submissions, including deferred Fusion worker execution.
pub(crate) struct KernelCounter {
    launches: AtomicUsize,
}

impl KernelCounter {
    pub(crate) fn start() -> (Arc<Self>, LaunchObservation) {
        let counter = Arc::new(Self {
            launches: AtomicUsize::new(0),
        });
        let observation = LaunchObservation::new(counter.clone());
        (counter, observation)
    }

    pub(crate) fn get(&self) -> usize {
        self.launches.load(Ordering::Relaxed)
    }
}

impl LaunchObserver for KernelCounter {
    fn launched(&self, kernel: &'static str) {
        if kernel.contains("optim::adamw::eager::adamw::") {
            self.launches.fetch_add(1, Ordering::Relaxed);
        }
    }
}
