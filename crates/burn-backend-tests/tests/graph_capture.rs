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

    /// A graph's output survives the replay of a graph captured before it on the device. The
    /// earlier graph's temporaries, the products of a chain of matmuls, replay over memory the
    /// device's graphs share; the later graph returns a tensor its closure allocates, which the
    /// capture keeps off that memory, so it reads what its own replay wrote whatever replays
    /// after it.
    #[test]
    #[serial]
    fn an_output_survives_an_earlier_graphs_replay() {
        const SIDE: usize = 1024;
        let device = Device::default();
        // Each entry 1/SIDE, so the matrix squares to itself and the chain keeps it.
        let wide_in = TestTensor::<2>::ones([SIDE, SIDE], &device).div_scalar(SIDE as f32);
        let mut wide = burn_tensor::capture(&device, || {
            let mut product = wide_in.clone();
            for _ in 0..4 {
                product = product.matmul(wide_in.clone());
            }
            product
        });
        let narrow_in = TestTensor::<2>::ones([SIDE, SIDE], &device);
        let mut narrow = burn_tensor::capture(&device, || narrow_in.clone().mul_scalar(3.0));
        assert!(wide.is_hardware(), "{:?}", wide.capture_error());
        assert!(narrow.is_hardware(), "{:?}", narrow.capture_error());

        // Safety: see `capture_replay_matches_eager`.
        let narrow_out = unsafe { narrow.replay() }.clone();
        let wide_out = unsafe { wide.replay() }.clone();
        narrow_out.into_data().assert_approx_eq::<FloatElem>(
            &TestTensor::<2>::full([SIDE, SIDE], 3.0, &device).into_data(),
            Tolerance::default(),
        );
        wide_out.into_data().assert_approx_eq::<FloatElem>(
            &TestTensor::<2>::full([SIDE, SIDE], 1.0 / SIDE as f32, &device).into_data(),
            Tolerance::default(),
        );
    }

    /// A graph's output survives the replay of a graph captured after it on the device, whose
    /// temporaries — the products of a chain of matmuls — take the memory the device's graphs
    /// share: the earlier output was kept off that memory when its own graph recorded.
    #[test]
    #[serial]
    fn an_output_survives_a_later_graphs_replay() {
        const SIDE: usize = 1024;
        let device = Device::default();
        let narrow_in = TestTensor::<2>::ones([SIDE, SIDE], &device);
        let mut narrow = burn_tensor::capture(&device, || narrow_in.clone().mul_scalar(3.0));
        let wide_in = TestTensor::<2>::ones([SIDE, SIDE], &device).div_scalar(SIDE as f32);
        let mut wide = burn_tensor::capture(&device, || {
            let mut product = wide_in.clone();
            for _ in 0..4 {
                product = product.matmul(wide_in.clone());
            }
            product
        });
        assert!(narrow.is_hardware(), "{:?}", narrow.capture_error());
        assert!(wide.is_hardware(), "{:?}", wide.capture_error());

        // Safety: see `capture_replay_matches_eager`.
        let narrow_out = unsafe { narrow.replay() }.clone();
        let wide_out = unsafe { wide.replay() }.clone();
        narrow_out.into_data().assert_approx_eq::<FloatElem>(
            &TestTensor::<2>::full([SIDE, SIDE], 3.0, &device).into_data(),
            Tolerance::default(),
        );
        wide_out.into_data().assert_approx_eq::<FloatElem>(
            &TestTensor::<2>::full([SIDE, SIDE], 1.0 / SIDE as f32, &device).into_data(),
            Tolerance::default(),
        );
    }

    /// Whether another clone of a buffer is still alive depends on the device's timing, so
    /// work that writes in place when run eagerly writes nothing in place while captured.
    #[test]
    #[serial]
    #[cfg(all(
        feature = "fusion",
        any(
            feature = "cuda",
            feature = "rocm",
            feature = "metal",
            feature = "wgpu"
        )
    ))]
    fn nothing_is_written_in_place_while_capturing() {
        use burn_cubecl_fusion::inspect::inplace_alias_count;

        let device = Device::default();
        let work = || {
            let fresh = TestTensor::<2>::ones([64, 64], &device);
            device.flush().unwrap();
            fresh.add_scalar(1.0).mul_scalar(2.0)
        };
        // The first run may tune, and a tuning trial never writes in place.
        let _ = work().into_data();

        let eager = inplace_alias_count();
        let _ = work().into_data();
        assert!(
            inplace_alias_count() > eager,
            "the work writes in place when run eagerly"
        );

        let captured = inplace_alias_count();
        let graph = burn_tensor::capture(&device, work);
        assert!(graph.is_hardware(), "{:?}", graph.capture_error());
        assert_eq!(inplace_alias_count(), captured);
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

        // Each run asks for a size no earlier run allocated, so the planning runs
        // never settle and the recorded run allocates otherwise than the last one.
        // The output the graph hands back then comes from a run that executed.
        let graph = burn_tensor::capture(&device, || {
            runs += 1;
            TestTensor::<1>::ones([786_433 * runs], &device)
        });

        assert!(!graph.is_hardware());
        assert!(graph.capture_error().is_some());
        let output = graph.output().clone();
        let ones = TestTensor::<1>::ones(output.shape(), &device);
        output
            .into_data()
            .assert_approx_eq::<FloatElem>(&ones.into_data(), Tolerance::default());
    }
}
