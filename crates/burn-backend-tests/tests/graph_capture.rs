//! Graph capture/replay integration tests for the closure-based
//! [`capture`](burn_tensor::capture) API.
//!
//! On a backend with hardware graph support (CUDA/HIP) the capture records the
//! closure's launches and every replay is a single dispatch against the
//! original buffers. Wgpu replays a software graph; backends without graph
//! support fall back to re-running the closure. All paths must produce the
//! eager result.
//!
//! Isolated in this test binary because CUDA's global capture mode can restrict
//! API calls from other threads during capture. Tests are additionally
//! `#[serial]` so their capture windows never overlap.

extern crate alloc;

#[cfg(feature = "cube")]
pub type FloatElem = f32;
#[cfg(feature = "cube")]
#[allow(unused)]
pub type IntElem = i32;

#[cfg(feature = "cube")]
#[path = "common/backend.rs"]
mod backend;

#[cfg(feature = "cube")]
mod cube {
    use super::{FloatElem, backend::*};

    use burn_tensor::{Device, Tolerance};
    use serial_test::serial;

    /// Replaying a captured pure closure reproduces the eager result, repeatedly.
    ///
    /// Safety of the `replay` calls: the closure owns a clone of `input` (keeping
    /// every captured buffer alive as long as the graph), and all replays and
    /// output reads happen sequentially on this thread — nothing else touches the
    /// graph's tensors.
    #[test]
    #[serial]
    fn capture_replay_matches_eager() {
        let device = Device::default();
        let input = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let expected = input.clone().mul_scalar(2.0).add_scalar(1.0).into_data();

        let mut graph =
            burn_tensor::capture(&device, || input.clone().mul_scalar(2.0).add_scalar(1.0));

        for _ in 0..3 {
            let out = unsafe { graph.replay() }.clone().into_data();
            out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
        }
    }

    /// The output handle is stable across replays on the hardware path: `output()`
    /// returns the same tensor whose buffer each replay overwrites.
    #[test]
    #[serial]
    fn capture_output_is_stable() {
        let device = Device::default();
        let input = TestTensor::<1>::from_data([1.0, 2.0, 3.0, 4.0], &device);
        let expected = input.clone().add_scalar(10.0).into_data();

        let mut graph = burn_tensor::capture(&device, || input.clone().add_scalar(10.0));

        // Safety: see `capture_replay_matches_eager`.
        unsafe { graph.replay() };
        graph
            .output()
            .clone()
            .into_data()
            .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
    }

    /// A closure that grows the memory pool inside the capture window cannot be
    /// recorded, since every replay must reuse the buffers prepared before it:
    /// the graph falls back to re-running the closure and says why. A backend
    /// without graph support fails at the start instead, so the assertions hold
    /// there too.
    #[test]
    #[serial]
    fn capture_reports_why_it_falls_back() {
        let device = Device::default();
        let mut runs = 0;

        // Each run asks for a size no earlier run allocated, so the captured run
        // cannot reuse a warmup buffer and must grow the pool.
        let graph = burn_tensor::capture(&device, || {
            runs += 1;
            TestTensor::<1>::ones([786_433 * runs], &device)
        });

        assert!(!graph.is_hardware());
        assert!(graph.capture_error().is_some());
    }
}
