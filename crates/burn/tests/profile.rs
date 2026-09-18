//! Profiling windows on a device.
//!
//! What these defend is the window's contract: it measures the work a closure
//! puts on the stream, in device time, without waiting for it — so windows
//! nest — and it closes where the stream stands when the closure returns, so
//! work put on the stream afterwards never counts, however late the work
//! inside it runs. On a backend that batches operations, the closure's last
//! operations can still be queued at that point; flushing forces them out
//! before the window closes.
//!
//! The batching case needs the fusion backend on a cubecl runtime:
//!
//! ```sh
//! cargo test -p burn --features vulkan --test profile
//! ```

#[cfg(any(
    feature = "cpu",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu"
))]
mod cube {
    use burn::prelude::{Device, Tensor};
    use burn::tensor::{ProfileDuration, ProfileOptions, ProfileTicks};
    use core::time::Duration;
    use std::sync::{Mutex, MutexGuard};

    /// One test on the device at a time. The device's runner thread is shared,
    /// and a runtime that stamps the stream (CUDA, HIP) measures the time it
    /// spends on another test's compile or tuning as part of whichever window
    /// happens to be open.
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

    /// Exclusive use of the device, kept even if another test panicked while
    /// holding it — the poison says nothing about the device itself.
    pub(super) fn one_at_a_time() -> MutexGuard<'static, ()> {
        ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn device() -> (MutexGuard<'static, ()>, Device) {
        let guard = one_at_a_time();

        #[cfg(feature = "cuda")]
        let device = Device::cuda(burn::tensor::DeviceIndex::Default);
        #[cfg(all(feature = "rocm", not(feature = "cuda")))]
        let device = Device::rocm(burn::tensor::DeviceIndex::Default);
        #[cfg(all(feature = "vulkan", not(any(feature = "cuda", feature = "rocm"))))]
        let device = Device::vulkan(burn::tensor::DeviceKind::DefaultDevice);
        #[cfg(all(
            feature = "wgpu",
            not(any(feature = "cuda", feature = "rocm", feature = "vulkan"))
        ))]
        let device = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
        #[cfg(all(
            feature = "cpu",
            not(any(
                feature = "cuda",
                feature = "rocm",
                feature = "vulkan",
                feature = "wgpu"
            ))
        ))]
        let device = Device::cpu();

        (guard, device)
    }

    fn resolve(duration: ProfileDuration) -> ProfileTicks {
        futures_lite::future::block_on(duration.resolve()).expect("the window carried work")
    }

    /// The element-wise median of three runs of `measure`.
    ///
    /// The tests that compare separately-scheduled windows are asserting a
    /// structural property — that the flush moved the work — off durations
    /// whose noise, on a loaded machine or a shared GPU, is the same order as
    /// the difference itself. One sample makes that a coin flip, and a
    /// coin-flip failure says nothing about what regressed. Three and a median
    /// is not a benchmark; it is what keeps the property from being decided by
    /// one scheduling spike.
    pub(super) fn median_of_three<const N: usize>(
        mut measure: impl FnMut() -> [Duration; N],
    ) -> [Duration; N] {
        let runs = [measure(), measure(), measure()];
        let mut median = [Duration::ZERO; N];
        for (at, slot) in median.iter_mut().enumerate() {
            let mut column = [runs[0][at], runs[1][at], runs[2][at]];
            column.sort();
            *slot = column[1];
        }
        median
    }

    /// `rounds` of work the device cannot skip, each a matmul the next
    /// depends on.
    fn work(device: &Device, rounds: usize) -> Tensor<2> {
        let mut x = Tensor::<2>::ones([512, 512], device);
        for _ in 0..rounds {
            x = x.clone().matmul(x) / 512.0;
        }
        x
    }

    /// A chain of element-wise operations, which the fusion backend holds in
    /// its queue as one block until something forces it out. Its output is
    /// large enough that writing it stands out from the noise of a window —
    /// a stream-stamping runtime measures host gaps across an idle one.
    fn lazy_chain(device: &Device) -> Tensor<1> {
        let mut x = Tensor::<1>::ones([32 * 1024 * 1024], device);
        for _ in 0..12 {
            x = (x * 1.5 + 0.5).exp().log();
        }
        x
    }

    #[test]
    fn read_inside_window_is_measured() {
        let (_guard, device) = device();

        let (sum, duration) = device
            .profile(|| work(&device, 8).sum().into_scalar::<f32>())
            .unwrap();

        assert!(sum.is_finite());
        assert!(resolve(duration).duration() > Duration::ZERO);
    }

    #[test]
    fn windows_nest() {
        let (_guard, device) = device();

        let ((_, inner), outer) = device
            .profile(|| {
                let a = work(&device, 8);
                let inner = device
                    .profile(|| work(&device, 8).sum().into_scalar::<f32>())
                    .unwrap();
                let _ = a.sum().into_scalar::<f32>();
                inner
            })
            .unwrap();

        let inner = resolve(inner).duration();
        let outer = resolve(outer).duration();
        assert!(inner > Duration::ZERO);
        assert!(inner <= outer, "inner {inner:?} exceeds outer {outer:?}");
    }

    /// Whether or not the closure's own work is still queued when the window
    /// closes, the far larger work put on the stream afterwards stays out.
    #[test]
    fn later_work_stays_out() {
        let (_guard, device) = device();

        // A window holds the compile and tuning of the kernels it launches
        // first, so the shapes are warmed before either window is compared.
        let _ = (work(&device, 32) + work(&device, 2))
            .sum()
            .into_scalar::<f32>();

        let (x, closed_early) = device.profile(|| work(&device, 2)).unwrap();
        let after = work(&device, 32) + x;
        let _ = after.sum().into_scalar::<f32>();

        let (_, closed_late) = device
            .profile(|| {
                let x = work(&device, 2);
                (work(&device, 32) + x).sum().into_scalar::<f32>()
            })
            .unwrap();

        let early = resolve(closed_early).duration();
        let late = resolve(closed_late).duration();
        assert!(
            early < late,
            "early {early:?} not below late {late:?}, so later work leaked in"
        );
    }

    /// What a flush guarantees is that the closure's work has run by the time
    /// the window closes. So the chain moves: out of the window that reads
    /// its output next, into the flushed window itself.
    ///
    /// The flushed window is not compared to an unflushed one directly: what
    /// an unflushed window holds is the queue's decision, and on a runtime
    /// that stamps the stream (CUDA, HIP) an idle window measures the host's
    /// registration time, noise of the same order as the chain. The reads are
    /// the yardstick instead — same work, same host shape, and only the chain
    /// between them.
    #[test]
    fn flush_runs_the_queue_inside_the_window() {
        let (_guard, device) = device();

        // Compiled before the windows are compared: see `later_work_stays_out`.
        let _ = lazy_chain(&device).sum().into_scalar::<f32>();

        let [flushed, read_after_lazy, read_after_flushed] = median_of_three(|| {
            let (x, _) = device.profile(|| lazy_chain(&device)).unwrap();
            let (_, read_after_lazy) = device.profile(|| x.sum().into_scalar::<f32>()).unwrap();

            let (x, flushed) = device
                .profile_with(ProfileOptions::default().flush(), || lazy_chain(&device))
                .unwrap();
            let (_, read_after_flushed) = device.profile(|| x.sum().into_scalar::<f32>()).unwrap();

            [
                resolve(flushed).duration(),
                resolve(read_after_lazy).duration(),
                resolve(read_after_flushed).duration(),
            ]
        });

        // A window nothing ran in reads as no time on a runtime that stamps
        // kernels (wgpu).
        assert!(flushed > Duration::ZERO, "the flushed window holds nothing");
        // What the flush took out of the read went into the flushed window.
        // Half of it, for the noise between two windows over the same work.
        assert!(
            read_after_lazy >= read_after_flushed + flushed / 2,
            "read after lazy {read_after_lazy:?}, read after flushed {read_after_flushed:?}, \
             flushed {flushed:?}: the flush did not move the chain into its window"
        );
    }

    /// Nothing holds the device while the closure runs, so a closure that
    /// waits on another thread's work on the same device — a data loader
    /// building its batch there, say — gets it back.
    #[test]
    fn closure_may_wait_on_another_thread() {
        let (_guard, device) = device();

        let (sum, _) = device
            .profile(|| {
                std::thread::scope(|scope| {
                    scope
                        .spawn(|| {
                            Tensor::<1>::ones([1024], &device)
                                .sum()
                                .into_scalar::<f32>()
                        })
                        .join()
                        .unwrap()
                })
            })
            .unwrap();

        assert_eq!(sum, 1024.0);
    }

    /// A panic out of the measured closure leaves no window open behind it.
    ///
    /// An open window is not free — a held start event, timestamp writes left
    /// on, retained command buffers — and on wgpu every later pass keeps
    /// rewriting the live window's end slot, so one leaked window taxes
    /// everything that runs afterwards. Nothing closes it later: the whole
    /// point of the split pair is that the server holds the window until a
    /// caller ends or abandons it.
    ///
    /// The leak is invisible from outside, so what this checks is that the
    /// device still measures afterwards: a window left open on wgpu takes the
    /// query set the next one needs, and a following profile comes back
    /// unmeasured or wrong.
    #[test]
    fn a_panicking_closure_abandons_its_window() {
        let (_guard, device) = device();

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = device.profile(|| {
                let _ = work(&device, 2);
                panic!("the measured work failed");
            });
        }));
        assert!(panicked.is_err(), "the panic reaches the caller");

        // The device is still profilable: the abandoned window released
        // whatever it was holding.
        let (sum, duration) = device
            .profile(|| work(&device, 4).sum().into_scalar::<f32>())
            .expect("the device still profiles after an abandoned window");

        assert!(sum.is_finite(), "the work after the panic still ran");
        assert!(
            resolve(duration).duration() > Duration::ZERO,
            "and it is still measured"
        );
    }

    /// An empty window is answered, not refused — and the output comes back
    /// whichever way the runtime answers.
    ///
    /// A runtime that stamps the stream (CUDA, HIP) reads the host gap between
    /// two stamps and reports about nothing. One that stamps kernels (wgpu)
    /// has no query set to read and reports **no measurement**, which is not a
    /// zero: it cannot tell a window that dispatched nothing from one whose
    /// work never reached a timestamped pass, and the second is a kernel that
    /// ran. So this asserts the shape both give — a window that resolves
    /// without erroring — rather than a duration only one of them has.
    #[test]
    fn empty_window_is_answered_either_way() {
        let (_guard, device) = device();

        let (out, duration) = device.profile(|| 42).unwrap();

        assert_eq!(out, 42);
        // Some(~0) on a stream-stamping runtime, None on a kernel-stamping
        // one. Both are answers; an error would not be.
        let _ = futures_lite::future::block_on(duration.resolve());
    }
}

#[test]
fn default_device_profiles_in_system_time() {
    use burn::prelude::{Device, Tensor};
    use burn::tensor::TimingMethod;

    // The default device is the cube tests' device under their features, and
    // this test's first-time compile would land in whichever window they had
    // open; so it takes its turn with them.
    #[cfg(any(
        feature = "cpu",
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan",
        feature = "wgpu"
    ))]
    let _guard = cube::one_at_a_time();

    let device = Device::default();
    let (sum, duration) = device
        .profile(|| {
            Tensor::<1>::ones([1024], &device)
                .sum()
                .into_scalar::<f32>()
        })
        .unwrap();

    assert_eq!(sum, 1024.0);
    if duration.timing_method() == TimingMethod::System {
        let ticks = futures_lite::future::block_on(duration.resolve()).unwrap();
        assert!(ticks.duration() > core::time::Duration::ZERO);
    }
}
