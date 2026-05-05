//! Tests that the fusion backend releases every tensor handle once the corresponding
//! [`FusionTensor`](burn_fusion::FusionTensor) wrappers are dropped; both for tensors
//! that live on a single stream and for tensors that are shared across streams.
//!
//! # Assertion Strategy
//! The assertion target is [`FusionInspector::new_handles_since_baseline`], which
//! returns every [`TensorId`](burn_ir::TensorId) that appeared in the
//! [`HandleContainer`](burn_ir::HandleContainer) *after* the baseline was set.
//!
//! Because the [`HandleContainer`](burn_ir::HandleContainer) is a global, per-device
//! registry, these tests call [`FusionInspector::enable_leak_detection`] to acquire
//! an exclusive lock on the device's handle state. This prevents concurrent tests
//! from allocating tensors that would otherwise appear as "leaks" in the baseline diff.
//!
//! # Threading and Concurrency
//! While execution reports are stream-isolated, memory handles are not. Therefore:
//! 1. These tests are isolated in this test binary to avoid false "leaks".
//! 2. Every test is marked `#[serial]` to ensure they take turns capturing the
//!    device-wide handle state.
//!
//! Cross-stream cases are simulated with [`StreamId::executes`], which swaps the
//! per-thread stream id for the duration of a closure. This is fast and deterministic;
//! the real OS-thread path is exercised by `tensor/multi_threads.rs`.

#![cfg(all(feature = "cube", feature = "fusion"))]

extern crate alloc;

pub type FloatElem = f32;
#[allow(unused)]
pub type IntElem = i32;

#[path = "common/backend.rs"]
mod backend;
pub use backend::*;

#[path = "fusion/mod.rs"]
mod fusion {

    use super::*;
    use burn_fusion::inspect::FusionInspector;
    use burn_tensor::{Device, StreamId, backend::Backend};
    use serial_test::serial;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Returns a unique `StreamId` for test isolation.
    pub fn test_stream() -> StreamId {
        static COUNTER: AtomicU64 = AtomicU64::new(1000);
        StreamId {
            value: COUNTER.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Drain both the main stream and `peer` stream so the post-sync handle snapshot
    /// reflects the full quiescent state.
    fn sync_all(device: &Device<TestBackend>, peer: StreamId) {
        TestBackend::sync(device).unwrap();
        peer.executes(|| TestBackend::sync(device).unwrap());
        TestBackend::sync(device).unwrap();
    }

    /// Drive the inspector into a known state, then freeze that state as the baseline.
    /// After this returns, [`FusionInspector::new_handles_since_baseline`] only reports
    /// handles born during the test body — handles from other parallel tests that were
    /// already live are excluded.
    fn install_and_baseline(device: &Device<TestBackend>, stream: StreamId) -> FusionInspector {
        let inspector = FusionInspector::install(stream);
        TestBackend::sync(device).unwrap();
        inspector.set_baseline();
        inspector
    }

    /// Assert that no handles registered during the test's window are still alive.
    #[track_caller]
    fn assert_no_leaked_handles(inspector: &FusionInspector, context: &str) {
        let leaked = inspector.new_handles_since_baseline();
        assert!(
            leaked.is_empty(),
            "{context}: {count} handle(s) registered during the test are still live: {leaked:?}",
            count = leaked.len(),
        );
    }

    /// Baseline: a tensor created and consumed on a single stream must not leave any
    /// handles behind.
    #[test]
    #[serial]
    fn single_stream_drop_frees_all_handles() {
        let stream = test_stream();
        stream.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, stream);

            {
                let a = TestTensor::<2>::ones([4, 4], &device);
                let b = TestTensor::<2>::ones([4, 4], &device);
                let _ = (a + b).into_data();
            }
            TestBackend::sync(&device).unwrap();

            assert_no_leaked_handles(&inspector, "single-stream drop + sync");
        });
    }

    /// `into_data` consumes its source — the source handle must be released when the
    /// stream is drained.
    #[test]
    #[serial]
    fn into_data_releases_source_handle() {
        let stream = test_stream();
        stream.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, stream);

            let a = TestTensor::<2>::ones([4, 4], &device);
            let _ = a.into_data();
            TestBackend::sync(&device).unwrap();

            assert_no_leaked_handles(&inspector, "into_data + sync");
        });
    }

    /// Cloning a tensor does not allocate a new handle; dropping the last `Arc`-style
    /// reference should release exactly one handle.
    #[test]
    #[serial]
    fn clone_then_drop_frees_handle_once() {
        let stream = test_stream();
        stream.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, stream);

            let a = TestTensor::<2>::ones([4, 4], &device);
            let b = a.clone();
            TestBackend::sync(&device).unwrap();

            drop(a);
            drop(b);
            TestBackend::sync(&device).unwrap();

            assert_no_leaked_handles(&inspector, "clone + drop + sync");
        });
    }

    /// Cross-stream sharing happy path: tensor created on the main stream, used and
    /// consumed on a peer stream, then dropped on the main stream. After every stream is
    /// drained, no handles must remain.
    #[test]
    #[serial]
    fn cross_stream_shared_tensor_released() {
        let stream = test_stream();
        let peer = test_stream();

        stream.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, stream);

            // Materialize `a` so its `Ones` init op doesn't get rolled into the cross-stream
            // analysis we're trying to observe.
            let a = TestTensor::<2>::ones([4, 4], &device);
            TestBackend::sync(&device).unwrap();

            // Performing an op on `a` from peer stream is what flags `a` as shared inside
            // `SharedTensors::analyse` (its creation stream is main, current is peer).
            let a_for_peer = a.clone();
            peer.executes(|| {
                let _ = (a_for_peer * 2.0).into_data();
            });

            drop(a);
            sync_all(&device, peer);

            assert_no_leaked_handles(
                &inspector,
                "shared tensor handle should be released once every sharing stream is drained",
            );
        });
    }

    /// The owner stream releases its reference *before* the peer stream catches up.
    /// The shared-tensor bookkeeping must defer the release until the peer drains; once
    /// it does, the handle must be reclaimed.
    #[test]
    #[serial]
    fn owner_drops_before_peer_finishes() {
        let stream = test_stream();
        let peer = test_stream();

        stream.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, stream);

            let a = TestTensor::<2>::ones([4, 4], &device);
            TestBackend::sync(&device).unwrap();

            // Queue work on the peer stream that depends on `a`, but do not drain yet so the
            // peer keeps a pending reference.
            let a_for_peer = a.clone();
            let peer_pending = peer.executes(|| a_for_peer * 2.0);

            // Owner releases its reference first, while peer still holds a pending op.
            drop(a);

            // Peer eventually catches up.
            peer.executes(|| {
                let _ = peer_pending.into_data();
            });

            sync_all(&device, peer);

            assert_no_leaked_handles(
                &inspector,
                "deferred shared-tensor drop should fire once the peer stream catches up",
            );
        });
    }

    /// The peer stream fully consumes its clone and closes *before* the owner drops its
    /// reference. The owner's later drop must still free the handle and not get stuck
    /// waiting on a stream that no longer exists.
    #[test]
    #[serial]
    fn peer_closes_before_owner_drops() {
        let stream = test_stream();
        let peer = test_stream();

        stream.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, stream);

            let a = TestTensor::<2>::ones([4, 4], &device);
            TestBackend::sync(&device).unwrap();

            let a_for_peer = a.clone();
            peer.executes(|| {
                let _ = (a_for_peer * 2.0).into_data();
            });
            // Peer is now closed: every var consumed, queue empty.

            drop(a);
            sync_all(&device, peer);

            assert_no_leaked_handles(
                &inspector,
                "owner-side drop after peer closure should still free the handle",
            );
        });
    }

    /// Stress: many iterations of cross-stream sharing back-to-back. The handle count
    /// must return to baseline at the end — catches regressions where each iteration
    /// leaks one or more handles (the failure mode the `drop-triggered drain` warning at
    /// `MultiStream::register` is meant to prevent).
    #[test]
    #[serial]
    fn cross_stream_loop_no_leak() {
        const REPS: usize = 8;
        let stream = test_stream();
        let peer = test_stream();

        stream.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, stream);

            for i in 0..REPS {
                let a = TestTensor::<2>::ones([4, 4], &device) * (i as f32);
                TestBackend::sync(&device).unwrap();

                let a_for_peer = a.clone();
                peer.executes(|| {
                    let _ = (a_for_peer + (i as f32)).into_data();
                });

                let _ = (a * 3.0).into_data();
            }

            sync_all(&device, peer);

            assert_no_leaked_handles(
                &inspector,
                "handle set must return to baseline after repeated cross-stream cleanup",
            );
        });
    }
}
