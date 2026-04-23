//! Tests that the fusion backend releases every tensor handle once the corresponding
//! [`FusionTensor`](burn_fusion::FusionTensor) wrappers are dropped — both for tensors
//! that live on a single stream and for tensors that are shared across streams.
//!
//! The assertion target is [`FusionInspector::new_handles_since_baseline`], which
//! returns every [`TensorId`](burn_ir::TensorId) that appeared in the
//! [`HandleContainer`](burn_ir::HandleContainer) *after* the baseline was set. The
//! [`HandleContainer`](burn_ir::HandleContainer) is shared per-device across the whole
//! process, so other tests running in parallel can add unrelated handles. Baselining
//! lets us diff against that noise and assert only on handles born during our test.
//! The inspector's install-mutex serializes inspector-based tests so their IDs do not
//! leak into the baseline window.
//!
//! Cross-stream cases are simulated with [`StreamId::executes`], which swaps the
//! per-thread stream id for the duration of a closure. This is fast and deterministic;
//! the real OS-thread path is exercised by `tensor/multi_threads.rs`.

use super::*;
use burn_fusion::inspect::FusionInspector;
use burn_tensor::{StreamId, backend::Backend};
use serial_test::serial;

/// Stream id used to play the role of a peer stream without spawning a real thread.
const PEER_STREAM: StreamId = StreamId { value: 0xdead_beef };

/// Drain both the main stream and [`PEER_STREAM`] so the post-sync handle snapshot
/// reflects the full quiescent state.
fn sync_all(device: &<TestBackend as Backend>::Device) {
    TestBackend::sync(device).unwrap();
    PEER_STREAM.executes(|| TestBackend::sync(device).unwrap());
    // A second main-stream sync ensures any drops triggered during peer drain are
    // flushed before we read the snapshot.
    TestBackend::sync(device).unwrap();
}

/// Drive the inspector into a known state, then freeze that state as the baseline.
/// After this returns, [`FusionInspector::new_handles_since_baseline`] only reports
/// handles born during the test body — handles from other parallel tests that were
/// already live are excluded.
fn install_and_baseline(device: &<TestBackend as Backend>::Device) -> FusionInspector {
    let inspector = FusionInspector::install();
    sync_all(device);
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
    let device = Default::default();
    let inspector = install_and_baseline(&device);

    {
        let a = TestTensor::<2>::ones([4, 4], &device);
        let b = TestTensor::<2>::ones([4, 4], &device);
        let _ = (a + b).into_data();
    }
    TestBackend::sync(&device).unwrap();

    assert_no_leaked_handles(&inspector, "single-stream drop + sync");
}

/// `into_data` consumes its source — the source handle must be released when the
/// stream is drained.
#[test]
#[serial]
fn into_data_releases_source_handle() {
    let device = Default::default();
    let inspector = install_and_baseline(&device);

    let a = TestTensor::<2>::ones([4, 4], &device);
    let _ = a.into_data();
    TestBackend::sync(&device).unwrap();

    assert_no_leaked_handles(&inspector, "into_data + sync");
}

/// Cloning a tensor does not allocate a new handle; dropping the last `Arc`-style
/// reference should release exactly one handle.
#[test]
#[serial]
fn clone_then_drop_frees_handle_once() {
    let device = Default::default();
    let inspector = install_and_baseline(&device);

    let a = TestTensor::<2>::ones([4, 4], &device);
    let b = a.clone();
    TestBackend::sync(&device).unwrap();

    drop(a);
    drop(b);
    TestBackend::sync(&device).unwrap();

    assert_no_leaked_handles(&inspector, "clone + drop + sync");
}

/// Cross-stream sharing happy path: tensor created on the main stream, used and
/// consumed on a peer stream, then dropped on the main stream. After every stream is
/// drained, no handles must remain.
#[test]
#[serial]
fn cross_stream_shared_tensor_released() {
    let device = Default::default();
    let inspector = install_and_baseline(&device);

    // Materialize `a` so its `Ones` init op doesn't get rolled into the cross-stream
    // analysis we're trying to observe.
    let a = TestTensor::<2>::ones([4, 4], &device);
    TestBackend::sync(&device).unwrap();

    // Performing an op on `a` from PEER_STREAM is what flags `a` as shared inside
    // `SharedTensors::analyse` (its creation stream is main, current is peer).
    let a_for_peer = a.clone();
    PEER_STREAM.executes(|| {
        let _ = (a_for_peer * 2.0).into_data();
    });

    drop(a);
    sync_all(&device);

    assert_no_leaked_handles(
        &inspector,
        "shared tensor handle should be released once every sharing stream is drained",
    );
}

/// The owner stream releases its reference *before* the peer stream catches up.
/// The shared-tensor bookkeeping must defer the release until the peer drains; once
/// it does, the handle must be reclaimed.
#[test]
#[serial]
fn owner_drops_before_peer_finishes() {
    let device = Default::default();
    let inspector = install_and_baseline(&device);

    let a = TestTensor::<2>::ones([4, 4], &device);
    TestBackend::sync(&device).unwrap();

    // Queue work on the peer stream that depends on `a`, but do not drain yet so the
    // peer keeps a pending reference.
    let a_for_peer = a.clone();
    let peer_pending = PEER_STREAM.executes(|| a_for_peer * 2.0);

    // Owner releases its reference first, while peer still holds a pending op.
    drop(a);

    // Peer eventually catches up.
    PEER_STREAM.executes(|| {
        let _ = peer_pending.into_data();
    });

    sync_all(&device);

    assert_no_leaked_handles(
        &inspector,
        "deferred shared-tensor drop should fire once the peer stream catches up",
    );
}

/// The peer stream fully consumes its clone and closes *before* the owner drops its
/// reference. The owner's later drop must still free the handle and not get stuck
/// waiting on a stream that no longer exists.
#[test]
#[serial]
fn peer_closes_before_owner_drops() {
    let device = Default::default();
    let inspector = install_and_baseline(&device);

    let a = TestTensor::<2>::ones([4, 4], &device);
    TestBackend::sync(&device).unwrap();

    let a_for_peer = a.clone();
    PEER_STREAM.executes(|| {
        let _ = (a_for_peer * 2.0).into_data();
    });
    // Peer is now closed: every var consumed, queue empty.

    drop(a);
    sync_all(&device);

    assert_no_leaked_handles(
        &inspector,
        "owner-side drop after peer closure should still free the handle",
    );
}

/// Stress: many iterations of cross-stream sharing back-to-back. The handle count
/// must return to baseline at the end — catches regressions where each iteration
/// leaks one or more handles (the failure mode the `drop-triggered drain` warning at
/// `MultiStream::register` is meant to prevent).
#[test]
#[serial]
fn cross_stream_loop_no_leak() {
    const REPS: usize = 8;
    let device = Default::default();
    let inspector = install_and_baseline(&device);

    for i in 0..REPS {
        let a = TestTensor::<2>::ones([4, 4], &device) * (i as f32);
        TestBackend::sync(&device).unwrap();

        let a_for_peer = a.clone();
        PEER_STREAM.executes(|| {
            let _ = (a_for_peer + (i as f32)).into_data();
        });

        let _ = (a * 3.0).into_data();
    }

    sync_all(&device);

    assert_no_leaked_handles(
        &inspector,
        "handle set must return to baseline after repeated cross-stream cleanup",
    );
}
