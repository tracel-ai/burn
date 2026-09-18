//! Profiling windows on a remote device, served in-process by the local cubecl
//! device.
//!
//! Its own test binary: hosting a server declares the whole process an async
//! runtime, which changes how the local device reads, and the local window
//! tests in `profile.rs` are not to run under it.
//!
//! ```sh
//! cargo test -p burn --features vulkan,remote-server,remote-websocket --test profile_remote
//! ```
#![cfg(all(
    feature = "remote-server",
    feature = "remote-websocket",
    any(
        feature = "cpu",
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan",
        feature = "wgpu"
    )
))]

use burn::prelude::{Device, Tensor};
use burn::tensor::{ProfileDuration, ProfileOptions};
use core::time::Duration;

/// What the window measured, or `None` where the device took no measurement.
///
/// An absence is not a zero, and the difference is the whole subject here: a
/// runtime that stamps kernels rather than the stream refuses a window that
/// nothing ran in, and the backend answers that refusal with no measurement
/// precisely so a caller comparing two windows cannot read the unmeasured one
/// as the quicker. The lazy window below is that window whenever the server
/// holds the chain back, so it is read as an absence rather than unwrapped.
fn measured(duration: ProfileDuration) -> Option<Duration> {
    futures_lite::future::block_on(duration.resolve()).map(|ticks| ticks.duration())
}

/// See `lazy_chain` in `profile.rs`: the server's fusion holds it in its
/// queue as one block until something forces it out.
fn lazy_chain(device: &Device) -> Tensor<1> {
    let mut x = Tensor::<1>::ones([32 * 1024 * 1024], device);
    for _ in 0..12 {
        x = (x * 1.5 + 0.5).exp().log();
    }
    x
}

/// A remote device's windows open on the server's backend, whose fusion
/// holds the closure's last operations back just as a local one does; the
/// flush has to travel with the close and drain that queue, not only the
/// client's.
///
/// Compared directly to an unflushed window, unlike the local test: over
/// the wire the chain dwarfs the host gap an idle window measures on a
/// runtime that stamps the stream, and one that stamps kernels reads an
/// idle window as no time.
#[test]
fn flush_reaches_the_server_queue() {
    let port = 3190;
    std::thread::spawn(move || {
        burn::server::start(Device::default(), burn::server::Channel::WebSocket { port })
    });
    std::thread::sleep(Duration::from_millis(500));

    let device = Device::remote_websocket(&format!("ws://localhost:{port}"), 0);

    // Compiled on the server before the windows are compared.
    let _ = lazy_chain(&device).sum().into_scalar::<f32>();

    // Three runs and the median of each: the ratio is large, but one
    // scheduling spike over the wire is not, and a flake here says nothing
    // about what regressed.
    let measure = || {
        let (x, lazy) = device.profile(|| lazy_chain(&device)).unwrap();
        let _ = x.sum().into_scalar::<f32>();

        let (x, flushed) = device
            .profile_with(ProfileOptions::default().flush(), || lazy_chain(&device))
            .unwrap();
        let _ = x.sum().into_scalar::<f32>();

        let flushed = measured(flushed)
            .expect("the flush ran the server's queue inside the window, leaving work to stamp");

        // No measurement at all is the strongest form of "the chain stayed
        // out", so it counts as no time rather than failing the run.
        (measured(lazy).unwrap_or(Duration::ZERO), flushed)
    };
    let mut runs = [measure(), measure(), measure()];
    runs.sort_by_key(|(_, flushed)| *flushed);
    let (lazy, flushed) = runs[1];
    assert!(
        flushed > lazy * 2,
        "flushed {flushed:?}, lazy {lazy:?}: the flush did not run the server's queue inside \
         its window"
    );
}
