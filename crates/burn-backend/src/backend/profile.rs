use burn_std::profile::Instant;
pub use burn_std::profile::{ProfileDuration, ProfileTicks, TimingMethod};

/// How a [profiled window](crate::Backend::profile) treats the work a backend
/// still holds in a queue when the window closes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProfileOptions {
    flush: bool,
}

impl ProfileOptions {
    /// Execute everything queued before closing the window, so the
    /// measurement holds all the work the closure registered.
    ///
    /// A backend that batches operations (fusion) otherwise leaves the
    /// closure's last operations queued past the window, and cuts nothing
    /// short — its batching is what production runs. Flushing forces the batch
    /// out at the window's end, which measures the closure's whole work at the
    /// cost of running it as production would not.
    pub fn flush(mut self) -> Self {
        self.flush = true;
        self
    }

    /// Whether the window flushes queued work before closing.
    pub fn flushes(&self) -> bool {
        self.flush
    }
}

/// An open profiling window on a backend, handed back by
/// [`profile_end`](crate::Backend::profile_end) to close it.
///
/// Windows nest and overlap: a backend keys each on its token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ProfileToken {
    /// The backend's own identifier for the window.
    pub id: u64,
}

/// Measure `func` in wall-clock time between two syncs of `device`.
///
/// What a backend with no device clock reports: the default of
/// [`Backend::profile`](crate::Backend::profile), and what a backend that
/// forwards its windows falls back to when the backend behind it opens none.
/// The syncs are what make the number mean something, so unlike a device
/// window this waits, and an inner window's syncs are charged to the outer.
pub fn profile_system_time<B: crate::Backend, O: Send + 'static>(
    device: &B::Device,
    func: impl FnOnce() -> O + Send,
) -> Result<(O, ProfileDuration), crate::ExecutionError> {
    B::sync(device)?;
    let start = Instant::now();
    let out = func();
    B::sync(device)?;
    Ok((out, ProfileDuration::new_system_time(start, Instant::now())))
}
