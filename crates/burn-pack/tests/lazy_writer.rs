//! Writing a container from tensor entries that materialize their bytes on demand.

mod common;

use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;

use burn_pack::{Bytes, DType, Error, Shape, Tensor, TensorEntry, Writer};
use common::f32_tensor;

/// Records every materialization, in the order it happened.
type Log = Rc<RefCell<Vec<String>>>;

/// A tensor entry that reports its byte length up front but only hands over the bytes when
/// the writer asks for them, logging each time it does.
struct LazyEntry {
    name: String,
    shape: Shape,
    values: Vec<f32>,
    /// Byte length as reported to the writer. Defaults to the real length; the consistency
    /// test overrides it to check the writer's guard.
    declared_len: usize,
    /// Set by the failure test to make [`TensorEntry::into_bytes`] error.
    fails: bool,
    log: Log,
}

impl LazyEntry {
    fn new(name: &str, shape: &[usize], first_value: f32, log: &Log) -> Self {
        let elements: usize = shape.iter().product();
        let values = (0..elements).map(|i| first_value + i as f32).collect();
        Self {
            name: name.to_string(),
            shape: Shape::from(shape.to_vec()),
            values,
            declared_len: elements * 4,
            fails: false,
            log: log.clone(),
        }
    }
}

impl TensorEntry for LazyEntry {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn dtype(&self) -> DType {
        DType::F32
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn param_id(&self) -> Option<u64> {
        None
    }

    fn byte_len(&self) -> usize {
        self.declared_len
    }

    fn into_bytes(self) -> Result<Bytes, Error> {
        self.log.borrow_mut().push(self.name.clone());
        if self.fails {
            return Err(Error::IoError("device read failed".to_string()));
        }
        let raw: Vec<u8> = self.values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(Bytes::from_bytes_vec(raw))
    }
}

fn entries(log: &Log) -> Vec<LazyEntry> {
    vec![
        LazyEntry::new("a", &[2, 2], 1.0, log),
        LazyEntry::new("b", &[3], 100.0, log),
        LazyEntry::new("c", &[64], 1000.0, log),
    ]
}

#[test]
fn planning_the_layout_materializes_nothing() {
    let log: Log = Rc::default();
    let writer = Writer::new(entries(&log));

    // `size()` builds the descriptors and the whole offset table.
    let size = writer.size().unwrap();
    assert!(size > 0);

    assert!(
        log.borrow().is_empty(),
        "computing the layout must not touch any tensor's bytes, materialized: {:?}",
        log.borrow()
    );
}

#[test]
fn each_tensor_materializes_once_in_write_order() {
    let log: Log = Rc::default();
    Writer::new(entries(&log)).into_bytes().unwrap();

    assert_eq!(log.borrow().as_slice(), ["a", "b", "c"]);
}

/// Byte-for-byte equality with the eager path is the strongest correctness statement
/// available here: it pins the lazy writer to behaviour `round_trip.rs` already covers.
#[test]
fn lazy_and_eager_entries_produce_identical_containers() {
    let log: Log = Rc::default();
    let lazy = entries(&log);

    let eager: Vec<Tensor> = lazy
        .iter()
        .map(|e| f32_tensor(&e.name, &e.values, &e.shape.to_vec(), None))
        .collect();

    let from_lazy = Writer::new(lazy).into_bytes().unwrap();
    let from_eager = Writer::new(eager).into_bytes().unwrap();

    assert_eq!(&from_lazy[..], &from_eager[..]);
}

#[test]
fn declared_length_that_disagrees_with_the_bytes_is_rejected() {
    let log: Log = Rc::default();
    let mut entry = LazyEntry::new("a", &[2, 2], 1.0, &log);
    // The layout reserves 8 bytes but the provider will hand back 16.
    entry.declared_len = 8;

    let err = Writer::new(vec![entry]).into_bytes().unwrap_err();
    match err {
        Error::TensorBytesSizeMismatch(msg) => {
            assert!(msg.contains("expected 8"), "unexpected message: {msg}");
            assert!(msg.contains("got 16"), "unexpected message: {msg}");
        }
        other => panic!("expected TensorBytesSizeMismatch, got {other:?}"),
    }
}

#[test]
fn a_failing_provider_aborts_the_write() {
    let log: Log = Rc::default();
    let mut entry = LazyEntry::new("a", &[1], 1.0, &log);
    entry.fails = true;

    let err = Writer::new(vec![entry]).into_bytes().unwrap_err();
    assert!(matches!(err, Error::IoError(msg) if msg.contains("device read failed")));
}
