pub use burn_std::profile::{ProfileDuration, ProfileTicks, TimingMethod};
use burn_std::{ExecutionError, backtrace::BackTrace, profile::Instant};

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

/// An open profiling window on a backend, handed out by
/// [`profile_start`](crate::Backend::profile_start) and passed back to
/// [`profile_end`](crate::Backend::profile_end) to close it.
///
/// Windows nest and overlap: a backend keys each on its token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ProfileToken {
    /// The backend's own identifier for the window.
    pub id: u64,
    /// The backend's own identifier for **where** the window was opened — a
    /// stream, a queue — carried so it can be closed there.
    ///
    /// Opaque to this crate, and `0` for a backend that has only one such
    /// place. A window belongs to where it was opened, not to whoever closes
    /// it: the split pair exists for work launched from somewhere else, so
    /// the two calls can land on different threads, and a close that guessed
    /// from the *calling* thread would pair a start recorded on one stream
    /// with an end recorded on another — which reads as a plausible duration
    /// and means nothing.
    #[serde(default)]
    pub opened_on: u64,
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

/// A closure-bracketed window over a backend's split
/// [`profile_start`](crate::Backend::profile_start) / [`profile_end`](crate::Backend::profile_end),
/// falling back to [`profile_system_time`] when the backend opens none.
///
/// What a forwarding backend measures with, when what it forwards to may or
/// may not have a device clock, and what a backend with a device clock
/// measures with when the closure must not run under a hold of the device.
pub fn profile_with_tokens<B: crate::Backend, O: Send + 'static>(
    device: &B::Device,
    options: ProfileOptions,
    func: impl FnOnce() -> O + Send,
) -> Result<(O, ProfileDuration), ExecutionError> {
    let opened = match B::profile_start(device) {
        Ok(Some(token)) => token,
        Ok(None) => return profile_system_time::<B, O>(device, func),
        // The window could not be opened — a remote device reached from a
        // browser thread, which cannot wait on the server. `func` still runs:
        // a caller asked for their work to be measured, and handing back an
        // error having quietly skipped the work is the one outcome they
        // cannot recover from. Measuring is what failed, so only the
        // measurement is lost.
        Err(err) => {
            func();
            return Err(err);
        }
    };

    // Held so an unwinding `func` abandons the window instead of leaving it
    // open on the server for the rest of the process.
    let mut window = OpenWindow::<B> {
        device,
        token: Some(opened),
    };
    let out = func();
    let token = window
        .token
        .take()
        .expect("the window is held from opening until here");
    let duration = B::profile_end(device, token, options)?;
    Ok((out, duration))
}

/// An open profiling window, abandoned on drop unless it was taken to be
/// closed. See [`crate::Backend::profile_abandon`].
struct OpenWindow<'a, B: crate::Backend> {
    device: &'a B::Device,
    token: Option<ProfileToken>,
}

impl<B: crate::Backend> Drop for OpenWindow<'_, B> {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            B::profile_abandon(self.device, token);
        }
    }
}

pub(crate) fn profile_unsupported() -> ExecutionError {
    ExecutionError::Generic {
        reason: alloc::string::String::from(
            "profiling windows are not supported by this backend; use `profile`",
        ),
        backtrace: BackTrace::capture(),
    }
}
