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

    /// Whether an operation writes in place is the same in every run of a capture, even when
    /// the device thread holds the buffer for a moment. Every other run of the closure lends the
    /// buffer it just allocated to a task on the device thread, which lets it go only after the
    /// operation that consumes it was submitted, as the device thread does with a handle it is
    /// still setting up. Deciding on the count of that moment would write in place in one run and
    /// allocate in the next, and the recorded run would allocate otherwise than the planning runs
    /// did.
    ///
    /// On the cube backend itself rather than through the fusion wrapper the device puts around
    /// it, so the tensor holds the buffer's handle.
    #[test]
    #[serial]
    fn an_in_place_write_does_not_depend_on_a_handle_the_device_thread_holds() {
        use burn_backend::{Scalar, TensorData, ops::FloatTensorOps};
        use burn_cubecl::CubeBackend;
        use std::sync::mpsc;
        use std::time::Duration;

        let device = Device::default();
        let burn_dispatch::DispatchDevice::Cube(device) = device.as_dispatch() else {
            return;
        };
        let input = CubeBackend::float_from_data(TensorData::from([1.0f32, 2.0, 3.0, 4.0]), device);
        let client = input.client.clone();
        let mut runs = 0;
        let captured = client.capture(|| {
            let doubled = CubeBackend::float_mul_scalar(input.clone(), Scalar::from(2.0f32));
            if runs % 2 == 1 {
                let held = doubled.clone();
                let client = held.client.clone();
                let (lent, on_device) = mpsc::channel();
                std::thread::spawn(move || {
                    client
                        .exclusive(move || {
                            lent.send(()).unwrap();
                            std::thread::sleep(Duration::from_millis(200));
                            drop(held);
                        })
                        .unwrap();
                });
                on_device.recv().unwrap();
            }
            runs += 1;
            CubeBackend::float_add_scalar(doubled, Scalar::from(1.0f32))
        });

        assert!(captured.is_ok(), "{:?}", captured.err());
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
