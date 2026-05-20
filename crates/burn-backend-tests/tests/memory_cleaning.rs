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
    use burn_tensor::{Device, StreamId, Transaction};
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
    fn sync_all(device: &Device, peer: StreamId) {
        device.sync().unwrap();
        peer.executes(|| device.sync().unwrap());
        device.sync().unwrap();
    }

    /// Drive the inspector into a known state, then freeze that state as the baseline.
    /// After this returns, [`FusionInspector::new_handles_since_baseline`] only reports
    /// handles born during the test body — handles from other parallel tests that were
    /// already live are excluded.
    fn install_and_baseline(device: &Device, stream: StreamId) -> FusionInspector {
        let inspector = FusionInspector::install(stream);
        device.sync().unwrap();
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
            device.sync().unwrap();

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
            device.sync().unwrap();

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
            device.sync().unwrap();

            drop(a);
            drop(b);
            device.sync().unwrap();

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
            device.sync().unwrap();

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
            device.sync().unwrap();

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
            device.sync().unwrap();

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
                device.sync().unwrap();

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

    /// Drain `extra` peer streams in addition to the main stream. Used by tests that
    /// share a single tensor across more than two streams.
    fn sync_streams(device: &Device, extra: &[StreamId]) {
        device.sync().unwrap();
        for s in extra {
            s.executes(|| device.sync().unwrap());
        }
        device.sync().unwrap();
    }

    /// A tensor created on the owner stream is consumed by three independent peer streams
    /// concurrently (no peer-to-peer chain). Every peer must observe the same data and the
    /// shared handle must be reclaimed after every stream has drained.
    #[test]
    #[serial]
    fn shared_tensor_three_peers_released() {
        let owner = test_stream();
        let p1 = test_stream();
        let p2 = test_stream();
        let p3 = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            device.sync().unwrap();

            // Each peer takes its own clone — each cross-stream `clone()` registers a
            // shared view aliasing `a`'s handle under a fresh tensor id.
            let c1 = a.clone();
            let c2 = a.clone();
            let c3 = a.clone();
            p1.executes(|| {
                let _ = (c1 * 2.0).into_data();
            });
            p2.executes(|| {
                let _ = (c2 + 1.0).into_data();
            });
            p3.executes(|| {
                let _ = (c3 - 1.0).into_data();
            });

            drop(a);
            sync_streams(&device, &[p1, p2, p3]);

            assert_no_leaked_handles(
                &inspector,
                "handle should be freed once every peer stream drains its alias",
            );
        });
    }

    /// Re-sharing path: tensor flows owner → peer → grandpeer. The grandpeer takes its
    /// clone from a tensor whose home stream is already `peer`, so the share path is
    /// triggered a second time from a different source stream.
    #[test]
    #[serial]
    fn chained_share_across_three_streams() {
        let owner = test_stream();
        let peer = test_stream();
        let grandpeer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            device.sync().unwrap();

            // First hop: owner -> peer. The clone happens on `owner` (regular clone),
            // then the multiplication on `peer` consumes the clone and triggers the
            // cross-stream `into_ir` path.
            let a_for_peer = a.clone();
            let on_peer = peer.executes(|| a_for_peer * 2.0);
            drop(a);

            // Second hop: peer -> grandpeer. `on_peer.stream == peer`; cloning while
            // current is grandpeer fires `shared_view` with `peer` as source.
            let on_peer_for_grand = grandpeer.executes(|| on_peer.clone());
            grandpeer.executes(|| {
                let _ = (on_peer_for_grand + 1.0).into_data();
            });

            // The peer tensor stays alive until here so peer's queue still references it.
            peer.executes(|| {
                let _ = on_peer.into_data();
            });

            sync_streams(&device, &[peer, grandpeer]);

            assert_no_leaked_handles(
                &inspector,
                "handle set must return to baseline after a chained owner→peer→grandpeer share",
            );
        });
    }

    /// Share a tensor that was *computed* on the owner stream rather than freshly
    /// initialised. This makes sure the drain-on-first-share path correctly flushes
    /// pending ops before exposing the handle to the peer.
    #[test]
    #[serial]
    fn share_computed_tensor_flushes_pending_ops() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            // Build a pending computation on owner without draining it.
            let a = TestTensor::<2>::ones([4, 4], &device);
            let b = (a.clone() + 1.0) * 3.0; // pending: (1 + 1) * 3 = 6
            // Intentionally do NOT sync — peer's first-share must drain the source.

            let b_for_peer = b.clone();
            let peer_data = peer.executes(|| (b_for_peer + 0.0).into_data());

            // 6 + 0 = 6
            peer_data.assert_approx_eq::<FloatElem>(
                &burn_tensor::TensorData::from([
                    [6.0_f32, 6.0, 6.0, 6.0],
                    [6.0, 6.0, 6.0, 6.0],
                    [6.0, 6.0, 6.0, 6.0],
                    [6.0, 6.0, 6.0, 6.0],
                ]),
                burn_tensor::Tolerance::default(),
            );

            drop(a);
            drop(b);
            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "computed-tensor share must flush pending ops and leave no handles behind",
            );
        });
    }

    /// Peer takes a clone but never executes any op on it before it is dropped.
    /// The aliased handle must still be reclaimed once the unused clone is freed.
    #[test]
    #[serial]
    fn share_then_drop_unused_clone() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            device.sync().unwrap();

            // Move the clone over to the peer stream but don't touch it.
            let unused = peer.executes(|| a.clone());
            peer.executes(|| drop(unused));

            drop(a);
            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "unused cross-stream clone must not leak its aliased handle",
            );
        });
    }

    /// Owner keeps using the tensor on its own stream after exposing a clone to the
    /// peer. Both sides must finish cleanly and the handle must be released only after
    /// the last reader drains.
    #[test]
    #[serial]
    fn owner_keeps_using_after_share() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            device.sync().unwrap();

            // Hand a clone to peer.
            let a_for_peer = a.clone();
            peer.executes(|| {
                let _ = (a_for_peer * 5.0).into_data();
            });

            // Owner continues doing work on `a` after the share point.
            let owner_result = (a.clone() + 2.0) * 3.0;
            let _ = owner_result.into_data();
            drop(a);

            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "owner must be able to keep using its tensor after sharing a clone",
            );
        });
    }

    /// Two distinct tensors are independently shared between the same pair of streams.
    /// Each share path must be tracked separately — dropping one must not affect the
    /// other.
    #[test]
    #[serial]
    fn multiple_distinct_shared_tensors_same_pair() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            let b = TestTensor::<2>::ones([4, 4], &device) * 2.0;
            device.sync().unwrap();

            let a_for_peer = a.clone();
            let b_for_peer = b.clone();

            peer.executes(|| {
                let _ = (a_for_peer + b_for_peer).into_data();
            });

            drop(a);
            drop(b);
            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "two independently shared tensors must each be released",
            );
        });
    }

    /// Int tensors travel through the same shared-view path as floats, but go through
    /// a different dtype branch in the runner. Cover that branch explicitly.
    #[test]
    #[serial]
    fn int_tensor_cross_stream_released() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensorInt::<1>::arange(0..16, &device);
            device.sync().unwrap();

            let a_for_peer = a.clone();
            peer.executes(|| {
                let _ = (a_for_peer + 1).into_data();
            });

            drop(a);
            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "int tensor sharing must release its aliased handle",
            );
        });
    }

    /// The peer performs several chained ops on the shared tensor. The shared handle
    /// is consumed (read-only) once on the peer side then dropped; we verify the full
    /// chain executes without leaks.
    #[test]
    #[serial]
    fn peer_runs_chain_on_shared_tensor() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            device.sync().unwrap();

            let a_for_peer = a.clone();
            peer.executes(|| {
                let x = a_for_peer * 2.0;
                let x = x + 1.0;
                let x = x * 4.0;
                let _ = x.into_data();
            });

            drop(a);
            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "multi-step chain on a shared tensor must not leak",
            );
        });
    }

    /// Owner sends a clone to peer; peer immediately sends a clone back to owner. The
    /// round trip exercises shared-view in both directions on a single underlying
    /// allocation.
    #[test]
    #[serial]
    fn share_round_trip_owner_peer_owner() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            device.sync().unwrap();

            // owner -> peer
            let a_for_peer = a.clone();
            let on_peer = peer.executes(|| a_for_peer + 1.0);

            // peer -> owner (cross-stream clone the other way)
            let back_to_owner = on_peer.clone();

            // Both sides consume their copy.
            let _ = (back_to_owner * 2.0).into_data();
            peer.executes(|| {
                let _ = on_peer.into_data();
            });

            drop(a);
            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "round-trip share owner→peer→owner must leave no handles behind",
            );
        });
    }

    /// `Tensor::into_data` consumes the tensor. When the consuming call happens on a
    /// stream different from the tensor's home stream, `into_ir` (not `clone`) must
    /// itself trigger the shared-view aliasing.
    #[test]
    #[serial]
    fn into_data_cross_stream_no_explicit_clone() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            device.sync().unwrap();

            // No explicit `.clone()` here — `a` is moved into the peer's closure and
            // consumed there. The cross-stream `into_ir` path must do the right thing.
            peer.executes(|| {
                let _ = a.into_data();
            });

            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "cross-stream into_ir must alias the handle without an explicit clone",
            );
        });
    }

    /// Same source tensor shared multiple times *to the same peer stream*. The
    /// first share triggers a drain of the source; subsequent shares of the same
    /// source must skip the drain (`MultiStream::resolved` short-circuit) but still
    /// register a fresh aliased handle.
    #[test]
    #[serial]
    fn repeated_share_same_source_same_peer() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            device.sync().unwrap();

            // Take three clones of the same `a`, all consumed on the peer stream.
            let c1 = a.clone();
            let c2 = a.clone();
            let c3 = a.clone();
            peer.executes(|| {
                let _ = (c1 + 1.0).into_data();
                let _ = (c2 + 2.0).into_data();
                let _ = (c3 + 3.0).into_data();
            });

            drop(a);
            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "repeated shares of the same source to the same peer must not leak",
            );
        });
    }

    /// The peer must observe the values the owner produced. Most other tests only
    /// check that handles are released; this one pins down data-side correctness across
    /// the shared-view aliasing boundary.
    #[test]
    #[serial]
    fn peer_observes_owner_values_after_share() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            // Owner stages a non-trivial pending computation: (1 + 1) * 5 = 10, then
            // shares the result with the peer *without* a manual sync. The first-share
            // drain must flush the chain so the peer sees 10, not garbage.
            let a = TestTensor::<2>::ones([2, 3], &device);
            let b = (a.clone() + 1.0) * 5.0;
            let b_for_peer = b.clone();

            let peer_data = peer.executes(|| (b_for_peer + 0.5).into_data());
            peer_data.assert_approx_eq::<FloatElem>(
                &burn_tensor::TensorData::from([[10.5_f32, 10.5, 10.5], [10.5, 10.5, 10.5]]),
                burn_tensor::Tolerance::default(),
            );

            // Owner still sees the same upstream values via its own copy of `b`.
            let owner_data = b.into_data();
            owner_data.assert_approx_eq::<FloatElem>(
                &burn_tensor::TensorData::from([[10.0_f32, 10.0, 10.0], [10.0, 10.0, 10.0]]),
                burn_tensor::Tolerance::default(),
            );

            drop(a);
            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "peer/owner views of the shared tensor must agree numerically and not leak",
            );
        });
    }

    /// Bool tensors take a separate dtype branch through `register_bool_tensor` and the
    /// boolean ops; sharing must route through the same path without leaks.
    #[test]
    #[serial]
    fn bool_tensor_cross_stream_released() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            // Build a bool tensor on owner: `a == a` is all-true.
            let mask = a.clone().equal(a.clone());
            device.sync().unwrap();

            let mask_for_peer = mask.clone();
            peer.executes(|| {
                let any_true = mask_for_peer.any();
                let data = any_true.into_data();
                let v: bool = data.iter::<bool>().next().expect("scalar bool");
                assert!(v, "shared bool tensor must remain all-true on peer side");
            });

            drop(a);
            drop(mask);
            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "bool tensor sharing must release its aliased handle",
            );
        });
    }

    /// A sliced view is a separate `FusionTensor` with its own id that depends on the
    /// parent's data. Sharing the view across streams must alias the *view's* handle
    /// (which is materialised by the slice op on the source stream) — not the parent's.
    #[test]
    #[serial]
    fn sliced_view_cross_stream_released() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::from_floats(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                &device,
            );

            // Slice on owner: 2x2 top-left block. The slice op produces a new tensor.
            let slice = a.clone().slice([0..2, 0..2]);
            let slice_for_peer = slice.clone();

            // Peer doubles every element of the view and reads it back.
            let peer_data = peer.executes(|| (slice_for_peer * 2.0).into_data());
            peer_data.assert_approx_eq::<FloatElem>(
                &burn_tensor::TensorData::from([[2.0_f32, 4.0], [10.0, 12.0]]),
                burn_tensor::Tolerance::default(),
            );

            drop(a);
            drop(slice);
            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "sliced-view sharing must release both parent and view aliases",
            );
        });
    }

    /// High fan-out: a single source tensor is shared with eight peer streams. This
    /// stresses the `MultiStream::resolved` short-circuit (one drain, many aliased
    /// handles) and the handle-removal path when every alias eventually drops.
    #[test]
    #[serial]
    fn high_fan_out_eight_peers_released() {
        let owner = test_stream();
        let peers: Vec<StreamId> = (0..8).map(|_| test_stream()).collect();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            device.sync().unwrap();

            // Each peer takes its own clone and consumes it.
            let clones: Vec<_> = (0..peers.len()).map(|_| a.clone()).collect();
            for (peer, c) in peers.iter().zip(clones) {
                let peer = *peer;
                peer.executes(|| {
                    let _ = (c + 1.0).into_data();
                });
            }

            drop(a);
            sync_streams(&device, &peers);

            assert_no_leaked_handles(
                &inspector,
                "many fan-out peers must each release their aliased handle",
            );
        });
    }

    /// Real OS-thread cross-stream sharing with leak detection. The other cross-stream
    /// tests use [`StreamId::executes`] to swap the per-thread id in place; this one
    /// actually spawns threads (the production code path) and checks that handles are
    /// reclaimed after every thread joins.
    #[test]
    #[serial]
    fn real_threads_cross_stream_no_leak() {
        let owner = test_stream();
        owner.executes(|| {
            let device: Device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            device.sync().unwrap();

            // Hand each worker thread its own clone. Cross-thread `clone()` is exactly
            // the path `FusionTensor::clone` takes in user code that spawns threads.
            let mut handles = Vec::new();
            for i in 0..4 {
                let c = a.clone();
                let dev = device.clone();
                handles.push(std::thread::spawn(move || {
                    let _ = (c + i as f32).into_data();
                    dev.sync().unwrap();
                }));
            }
            for h in handles {
                h.join().unwrap();
            }

            drop(a);
            device.sync().unwrap();

            assert_no_leaked_handles(
                &inspector,
                "real-thread cross-stream sharing must not leak handles",
            );
        });
    }

    /// Two peer streams *both* take a clone of the same source, and one of them
    /// re-shares its clone onward to a third stream. Exercises the case where a
    /// non-owner stream is itself the source of a `tag_shared_view` call.
    #[test]
    #[serial]
    fn peer_reshares_to_third_stream() {
        let owner = test_stream();
        let p1 = test_stream();
        let p2 = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            device.sync().unwrap();

            // owner -> p1 (consumed)
            let c1 = a.clone();
            let on_p1 = p1.executes(|| c1 + 1.0); // on_p1.stream == p1, value = 2

            // p1 -> p2 (re-share); the source stream for tag_shared_view is p1, not owner.
            let on_p1_clone = p2.executes(|| on_p1.clone());
            let p2_data = p2.executes(|| (on_p1_clone * 3.0).into_data());
            p2_data.assert_approx_eq::<FloatElem>(
                &burn_tensor::TensorData::from([
                    [6.0_f32, 6.0, 6.0, 6.0],
                    [6.0, 6.0, 6.0, 6.0],
                    [6.0, 6.0, 6.0, 6.0],
                    [6.0, 6.0, 6.0, 6.0],
                ]),
                burn_tensor::Tolerance::default(),
            );

            // Drain p1's pending tensor.
            p1.executes(|| {
                let _ = on_p1.into_data();
            });

            drop(a);
            sync_streams(&device, &[p1, p2]);

            assert_no_leaked_handles(
                &inspector,
                "peer-as-source share path must release every alias",
            );
        });
    }

    /// Both streams have pending (un-drained) work referencing the shared tensor when
    /// the top-level test syncs. The drain path on either stream must converge.
    #[test]
    #[serial]
    fn concurrent_pending_work_on_both_streams() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            let a = TestTensor::<2>::ones([4, 4], &device);
            device.sync().unwrap();

            // Build a pending op on peer, do not drain.
            let a_for_peer = a.clone();
            let peer_pending = peer.executes(|| a_for_peer * 2.0);

            // Build a pending op on owner, do not drain.
            let owner_pending = a.clone() + 5.0;

            // Now drop `a` and finalise both sides via a coordinated sync.
            drop(a);

            peer.executes(|| {
                let _ = peer_pending.into_data();
            });
            let _ = owner_pending.into_data();

            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "both streams holding pending work on a shared tensor must converge cleanly",
            );
        });
    }

    #[test]
    #[serial]
    fn transaction_cross_stream_no_reentrant_panic() {
        let owner = test_stream();
        let peer = test_stream();

        owner.executes(|| {
            let device = Default::default();
            let inspector = install_and_baseline(&device, owner);

            // Create and compute a tensor on the owner stream, leave it un-drained.
            let a = TestTensor::<2>::ones([4, 4], &device);
            let b = (a.clone() + 1.0) * 2.0;

            // Execute the transaction from the peer stream, `b.stream == owner` but
            // `current == peer`, so `resolve_tensor_float` hits the cross-stream mismatch
            // and previously called `into_ir()` inside the `submit_blocking` closure,
            // triggering the re-entrancy panic.
            let [data] = peer.executes(|| {
                Transaction::default()
                    .register(b)
                    .execute()
                    .try_into()
                    .unwrap()
            });

            data.assert_approx_eq::<FloatElem>(
                &burn_tensor::TensorData::from([
                    [4.0_f32, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0],
                ]),
                burn_tensor::Tolerance::default(),
            );

            drop(a);
            sync_streams(&device, &[peer]);

            assert_no_leaked_handles(
                &inspector,
                "transaction executed cross-stream must not panic or leak handles",
            );
        });
    }
}
