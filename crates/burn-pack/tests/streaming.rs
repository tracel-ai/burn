//! Streaming write (`write_to`) / read (`from_reader`) coverage.

mod common;

use burn_pack::{DType, Reader, Scalar, Tensor, Writer};
use common::{f32_tensor, read_f32};
use std::io::Read;

/// Reader that yields at most `max` bytes per `read` call, to stress the `read_exact` /
/// padding-skip loops against short reads (as a socket or pipe would produce).
struct ChunkedReader<'a> {
    data: &'a [u8],
    max: usize,
}

impl Read for ChunkedReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = self.data.len().min(out.len()).min(self.max);
        out[..n].copy_from_slice(&self.data[..n]);
        self.data = &self.data[n..];
        Ok(n)
    }
}

fn varied_tensors() -> Vec<Tensor> {
    // Odd element counts in write order (z, a, m) force alignment gaps between tensors.
    let z: Vec<f32> = (0..5).map(|i| 100.0 + i as f32).collect();
    let a: Vec<f32> = (0..3).map(|i| i as f32).collect();
    let m: Vec<f32> = (0..7).map(|i| 50.0 + i as f32).collect();
    vec![
        f32_tensor("zebra", &z, &[5], Some(1)),
        f32_tensor("alpha", &a, &[3], None),
        f32_tensor("mango", &m, &[7], Some(2)),
    ]
}

#[test]
fn write_to_matches_into_bytes() {
    let via_bytes = Writer::new(varied_tensors())
        .with_metadata("producer", "burn-pack")
        .with_scalar("step", Scalar::from(42u64))
        .into_bytes()
        .unwrap();

    let mut via_stream = Vec::new();
    Writer::new(varied_tensors())
        .with_metadata("producer", "burn-pack")
        .with_scalar("step", Scalar::from(42u64))
        .write_to(&mut via_stream)
        .unwrap();

    assert_eq!(via_stream.as_slice(), &via_bytes[..]);
}

#[test]
fn from_reader_round_trips_tensors() {
    let mut buf = Vec::new();
    Writer::new(varied_tensors()).write_to(&mut buf).unwrap();

    let tensors = Reader::from_reader(buf.as_slice())
        .unwrap()
        .into_tensors()
        .unwrap();

    let names: Vec<_> = tensors.iter().map(|t| t.name.clone()).collect();
    assert_eq!(names, vec!["alpha", "mango", "zebra"]);
    assert_eq!(read_f32(&tensors[0]), vec![0.0, 1.0, 2.0]);
    assert_eq!(read_f32(&tensors[1]), vec![50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0]);
    assert_eq!(read_f32(&tensors[2]), vec![100.0, 101.0, 102.0, 103.0, 104.0]);
    assert_eq!(tensors[2].param_id, Some(1));
    assert_eq!(tensors[0].param_id, None);
    assert_eq!(tensors[1].dtype, DType::F32);
}

#[test]
fn from_reader_reads_into_bytes_output() {
    // Cross-path: serialize with the buffer writer, deserialize with the stream reader.
    let packed = Writer::new(varied_tensors()).into_bytes().unwrap();
    let tensors = Reader::from_reader(&packed[..])
        .unwrap()
        .into_tensors()
        .unwrap();
    assert_eq!(tensors.len(), 3);
    assert_eq!(read_f32(&tensors[2]), vec![100.0, 101.0, 102.0, 103.0, 104.0]);
}

#[test]
fn from_reader_tolerates_short_reads() {
    let mut buf = Vec::new();
    Writer::new(varied_tensors()).write_to(&mut buf).unwrap();

    let reader = ChunkedReader {
        data: &buf,
        max: 1,
    };
    let tensors = Reader::from_reader(reader).unwrap().into_tensors().unwrap();
    assert_eq!(tensors.len(), 3);
    assert_eq!(read_f32(&tensors[0]), vec![0.0, 1.0, 2.0]);
}

#[test]
fn streaming_metadata_and_scalars_round_trip() {
    let mut buf = Vec::new();
    Writer::new(vec![f32_tensor("w", &[1.0], &[1], None)])
        .with_metadata("producer", "burn-pack")
        .with_scalar("epoch", Scalar::from(7u64))
        .with_scalar("lr", Scalar::from(0.5f64))
        .write_to(&mut buf)
        .unwrap();

    let reader = Reader::from_reader(buf.as_slice()).unwrap();
    assert_eq!(reader.metadata()["producer"], "burn-pack");
    assert_eq!(u64::try_from(reader.scalars()["epoch"]).unwrap(), 7);
    assert_eq!(f64::try_from(reader.scalars()["lr"]).unwrap(), 0.5);
    assert_eq!(reader.tensor_data("w").unwrap().len(), 4);
}

#[test]
fn from_reader_empty_pack() {
    let mut buf = Vec::new();
    Writer::new(vec![]).write_to(&mut buf).unwrap();

    let reader = Reader::from_reader(buf.as_slice()).unwrap();
    assert!(reader.tensor_names().is_empty());
    assert!(reader.into_tensors().unwrap().is_empty());
}
