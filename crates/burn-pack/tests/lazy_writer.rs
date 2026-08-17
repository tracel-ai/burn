//! Writing a container from tensors that produce their bytes on demand.

mod common;

use std::sync::{Arc, Mutex};

use burn_pack::{Bytes, DType, Error, Reader, Shape, Tensor, Writer};
use common::f32_tensor;

/// Records every materialization, in the order it happened.
///
/// `Arc<Mutex<_>>` rather than `Rc<RefCell<_>>` because a deferred tensor's provider must be
/// `Send + Sync`: records holding one cross threads through burn-train's async checkpointer.
type Log = Arc<Mutex<Vec<String>>>;

/// A tensor whose bytes are produced on demand, with every knob the tests need to bend.
///
/// Built into a [`Tensor`] by [`build`](Self::build); the fields stay reachable until then so
/// a test can contradict one of them (declare a length the shape cannot justify, produce
/// fewer values than declared) and watch the writer catch it.
struct LazyEntry {
    name: String,
    /// `F32` unless a test overrides it; the payload stays f32 either way, since the writer
    /// never interprets the bytes.
    dtype: DType,
    shape: Shape,
    /// The values the provider hands over. Normally matches `shape`; the mid-write guard test
    /// shortens it so the tensor produces less than it promised.
    values: Vec<f32>,
    /// Byte length as declared to the writer. Normally matches both `shape` and `values`; the
    /// plan-time guard test overrides it alone.
    declared_len: usize,
    /// Set by the failure tests to make the provider error.
    fails: bool,
    log: Log,
}

impl LazyEntry {
    fn new(name: &str, shape: impl Into<Shape>, first_value: f32, log: &Log) -> Self {
        let shape = shape.into();
        let elements = shape.num_elements();
        // A ramp from a per-tensor base, so every tensor's bytes are unique: the eager-vs-lazy
        // container comparison is byte-for-byte, and identical payloads would let swapped or
        // misplaced tensor data pass it unnoticed.
        let values = (0..elements).map(|i| first_value + i as f32).collect();
        Self {
            name: name.to_string(),
            dtype: DType::F32,
            shape,
            values,
            declared_len: elements * 4,
            fails: false,
            log: log.clone(),
        }
    }

    fn build(self) -> Tensor {
        let Self {
            name,
            dtype,
            shape,
            values,
            declared_len,
            fails,
            log,
        } = self;

        let logged = name.clone();
        Tensor::deferred(name, dtype, shape, None, declared_len, move || {
            log.lock().unwrap().push(logged.clone());
            if fails {
                return Err(Error::IoError("device read failed".to_string()));
            }
            let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok(Bytes::from_bytes_vec(raw))
        })
    }
}

fn entries(log: &Log) -> Vec<LazyEntry> {
    vec![
        LazyEntry::new("a", [2, 2], 1.0, log),
        LazyEntry::new("b", [3], 100.0, log),
        LazyEntry::new("c", [64], 1000.0, log),
    ]
}

fn deferred_tensors(entries: Vec<LazyEntry>) -> Vec<Tensor> {
    entries.into_iter().map(LazyEntry::build).collect()
}

fn materialized(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

#[test]
fn planning_the_layout_materializes_nothing() {
    let log = Log::default();
    let writer = Writer::new(deferred_tensors(entries(&log)));

    // `size()` builds the descriptors and the whole offset table.
    let size = writer.size().unwrap();
    assert!(size > 0);

    assert!(
        materialized(&log).is_empty(),
        "computing the layout must not touch any tensor's bytes, materialized: {:?}",
        materialized(&log)
    );
}

#[test]
fn each_tensor_materializes_once_in_write_order() {
    let log = Log::default();
    Writer::new(deferred_tensors(entries(&log)))
        .into_bytes()
        .unwrap();

    assert_eq!(materialized(&log), ["a", "b", "c"]);
}

/// Byte-for-byte equality with the eager path is the strongest correctness statement
/// available here: it pins the deferred writer to behaviour `round_trip.rs` already covers.
#[test]
fn deferred_and_resident_tensors_produce_identical_containers() {
    let log = Log::default();
    let lazy = entries(&log);

    let eager: Vec<Tensor> = lazy
        .iter()
        .map(|e| f32_tensor(&e.name, &e.values, &e.shape.to_vec(), None))
        .collect();

    let from_lazy = Writer::new(deferred_tensors(lazy)).into_bytes().unwrap();
    let from_eager = Writer::new(eager).into_bytes().unwrap();

    assert_eq!(&from_lazy[..], &from_eager[..]);
}

/// A `byte_len` that cannot describe the declared shape and dtype is caught while planning,
/// before any I/O happens, because the two are checked against each other directly.
#[test]
fn declared_length_that_contradicts_the_shape_is_rejected_before_writing() {
    let log = Log::default();
    let mut entry = LazyEntry::new("a", [2, 2], 1.0, &log);
    // A 2x2 f32 tensor needs 16 bytes, whatever the tensor claims.
    entry.declared_len = 8;

    let err = Writer::new(vec![entry.build()]).into_bytes().unwrap_err();
    match err {
        Error::ValidationError(msg) => {
            assert!(
                msg.contains("declares 8 bytes"),
                "unexpected message: {msg}"
            );
            assert!(msg.contains("need 16"), "unexpected message: {msg}");
        }
        other => panic!("expected ValidationError, got {other:?}"),
    }
    assert!(
        materialized(&log).is_empty(),
        "planning rejected the tensor, so nothing should have been materialized"
    );
}

/// Quantized tensors are exempt from the plan-time shape check: their length is not
/// `num_elements * dtype.size()` (packed values, inline scales), so a `byte_len` that would
/// be rejected for any other dtype must be accepted for `QFloat` - the exemption is what
/// makes [`Tensor::deferred`]'s explicit byte length usable for quantized data at all.
#[test]
fn a_quantized_byte_len_is_not_shape_checked_at_plan_time() {
    use burn_std::QuantScheme;

    let log = Log::default();
    let mut entry = LazyEntry::new("q", [3], 1.0, &log);
    entry.dtype = DType::QFloat(QuantScheme::default());
    // 3 elements could never need 8 bytes under `num_elements * size()` arithmetic; for
    // F32 this exact setup is rejected while planning (see the test above). The payload
    // matches the declaration, so the mid-write guard is satisfied too.
    entry.values.truncate(2);
    entry.declared_len = 8;

    Writer::new(vec![entry.build()]).into_bytes().unwrap();
}

/// A `byte_len` consistent with the shape but a provider that produces something else slips
/// past planning, so the writer re-checks the actual bytes before emitting them. This is the
/// only guard quantized tensors get, since their length is not a product of shape and dtype.
#[test]
fn provider_that_produces_a_different_length_is_rejected_mid_write() {
    let log = Log::default();
    let mut entry = LazyEntry::new("a", [2, 2], 1.0, &log);
    // `declared_len` stays 16 and agrees with the shape; the bytes do not.
    entry.values.truncate(2);

    let err = Writer::new(vec![entry.build()]).into_bytes().unwrap_err();
    match err {
        Error::TensorBytesSizeMismatch(msg) => {
            assert!(msg.contains("expected 16"), "unexpected message: {msg}");
            assert!(msg.contains("got 8"), "unexpected message: {msg}");
        }
        other => panic!("expected TensorBytesSizeMismatch, got {other:?}"),
    }
}

#[test]
fn a_failing_provider_aborts_the_write() {
    let log = Log::default();
    let mut entry = LazyEntry::new("a", [1], 1.0, &log);
    entry.fails = true;

    let err = Writer::new(vec![entry.build()]).into_bytes().unwrap_err();
    // The name prefix is the writer's `in_tensor` annotation: a provider failure arrives
    // mid-write, and without it the user cannot tell which of many tensors failed.
    assert!(
        matches!(&err, Error::IoError(msg) if msg.contains("tensor 'a'") && msg.contains("device read failed")),
        "expected a named IoError, got {err:?}"
    );
}

/// Planning rejects an inconsistent tensor before the scratch file is even created, so a
/// plan-time failure touches the disk not at all.
#[test]
fn a_plan_time_rejection_creates_no_file() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("model.bpk");

    let log = Log::default();
    let mut entry = LazyEntry::new("a", [2, 2], 1.0, &log);
    entry.declared_len = 8;

    Writer::new(vec![entry.build()])
        .write_to_file(&dest)
        .unwrap_err();

    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        0,
        "a plan-time rejection should create neither destination nor scratch file"
    );
}

/// A deferred provider runs during the write, so its failure has to leave the destination as
/// it was rather than truncated. Failing on the *last* tensor means the earlier ones were
/// already streamed out, which is exactly the case a plain `File::create` would corrupt.
#[test]
fn a_failing_provider_leaves_no_file_behind() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("model.bpk");

    let log = Log::default();
    let mut entries = entries(&log);
    entries.last_mut().unwrap().fails = true;

    Writer::new(deferred_tensors(entries))
        .write_to_file(&dest)
        .unwrap_err();

    assert!(!dest.exists(), "destination should not have been created");
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        0,
        "scratch file should have been cleaned up"
    );
}

#[test]
fn a_failed_write_leaves_an_existing_file_intact() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("model.bpk");

    // A valid container already at the destination.
    let log = Log::default();
    Writer::new(deferred_tensors(entries(&log)))
        .write_to_file(&dest)
        .unwrap();
    let original = std::fs::read(&dest).unwrap();

    let mut entries = entries(&log);
    entries.last_mut().unwrap().fails = true;
    Writer::new(deferred_tensors(entries))
        .write_to_file(&dest)
        .unwrap_err();

    assert_eq!(
        std::fs::read(&dest).unwrap(),
        original,
        "a failed write must not disturb the previous container"
    );
}

/// The other atomicity tests all fail before the rename. This one fails *at* it, leaving a
/// finished, full-size scratch file as the thing to clean up.
#[test]
fn a_failed_rename_leaves_no_scratch_file() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("model.bpk");
    // A directory cannot be replaced by a rename, so persist fails after a complete write.
    std::fs::create_dir(&dest).unwrap();

    let log = Log::default();
    Writer::new(deferred_tensors(entries(&log)))
        .write_to_file(&dest)
        .unwrap_err();

    assert!(dest.is_dir(), "the destination should be untouched");
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        1,
        "the completed scratch file should have been cleaned up"
    );
}

#[test]
fn a_successful_write_replaces_an_existing_file() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("model.bpk");

    let log = Log::default();
    Writer::new(vec![LazyEntry::new("old", [4], 1.0, &log).build()])
        .write_to_file(&dest)
        .unwrap();
    Writer::new(deferred_tensors(entries(&log)))
        .write_to_file(&dest)
        .unwrap();

    let names: Vec<String> = Reader::from_file(&dest)
        .unwrap()
        .into_tensors()
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(names, ["a", "b", "c"]);
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        1,
        "no scratch file should survive a successful write"
    );
}
