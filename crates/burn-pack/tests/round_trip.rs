//! Write → read round-trip coverage for the burnpack format.

mod common;

use burn_pack::{DType, Reader, Writer};
use common::{f32_tensor, raw_tensor, read_f32};

#[test]
fn single_tensor_round_trip() {
    let tensor = f32_tensor("weight", &[1.0, 2.0, 3.0, 4.0], &[2, 2], Some(7));

    let packed = Writer::new(vec![tensor]).to_bytes().unwrap();
    let reader = Reader::from_bytes(packed).unwrap();
    let tensors = reader.get_tensors().unwrap();

    assert_eq!(tensors.len(), 1);
    let t = &tensors[0];
    assert_eq!(t.name, "weight");
    assert_eq!(t.dtype, DType::F32);
    assert_eq!(t.shape.to_vec(), vec![2, 2]);
    assert_eq!(t.param_id, Some(7));
    assert_eq!(t.byte_len(), 16);
    assert_eq!(read_f32(t), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn multiple_tensors_returned_sorted_by_name() {
    // Insert out of order; the reader yields them sorted (BTreeMap ordering).
    let packed = Writer::new(vec![
        f32_tensor("zebra", &[9.0], &[1], None),
        f32_tensor("alpha", &[1.0], &[1], None),
        f32_tensor("mango", &[5.0], &[1], None),
    ])
    .to_bytes()
    .unwrap();

    let reader = Reader::from_bytes(packed).unwrap();
    let names: Vec<_> = reader
        .get_tensors()
        .unwrap()
        .iter()
        .map(|t| t.name.clone())
        .collect();

    assert_eq!(names, vec!["alpha", "mango", "zebra"]);
    assert_eq!(reader.tensor_names(), vec!["alpha", "mango", "zebra"]);
}

#[test]
fn user_metadata_round_trip() {
    let packed = Writer::new(vec![f32_tensor("w", &[1.0], &[1], None)])
        .with_metadata("producer", "burn-pack")
        .with_metadata("format", "burnpack")
        .to_bytes()
        .unwrap();

    let reader = Reader::from_bytes(packed).unwrap();
    assert_eq!(reader.metadata()["producer"], "burn-pack");
    assert_eq!(reader.metadata()["format"], "burnpack");
    // Per-tensor info comes from the tensors themselves.
    let tensors = reader.get_tensors().unwrap();
    assert_eq!(tensors[0].dtype, DType::F32);
    assert_eq!(tensors[0].shape.to_vec(), vec![1]);
}

#[test]
fn param_id_present_and_absent() {
    let packed = Writer::new(vec![
        f32_tensor("with_id", &[1.0], &[1], Some(123)),
        f32_tensor("without_id", &[2.0], &[1], None),
    ])
    .to_bytes()
    .unwrap();

    let reader = Reader::from_bytes(packed).unwrap();
    let tensors = reader.get_tensors().unwrap();
    let with = tensors.iter().find(|t| t.name == "with_id").unwrap();
    let without = tensors.iter().find(|t| t.name == "without_id").unwrap();

    assert_eq!(with.param_id, Some(123));
    assert_eq!(without.param_id, None);
}

#[test]
fn empty_pack() {
    let packed = Writer::new(vec![]).to_bytes().unwrap();
    let reader = Reader::from_bytes(packed).unwrap();
    assert!(reader.get_tensors().unwrap().is_empty());
    assert!(reader.tensor_names().is_empty());
}

#[test]
fn dtype_and_byte_len_preserved() {
    // (dtype, bytes-per-element)
    let cases = [
        (DType::F32, 4usize),
        (DType::F64, 8),
        (DType::I64, 8),
        (DType::I32, 4),
        (DType::I8, 1),
        (DType::U8, 1),
        (DType::BF16, 2),
        (DType::F16, 2),
    ];

    let n = 3;
    let tensors = cases
        .iter()
        .enumerate()
        .map(|(i, (dtype, elem))| {
            let bytes = vec![i as u8 + 1; n * elem];
            raw_tensor(&format!("t{i}"), *dtype, &[n], bytes, None)
        })
        .collect();

    let packed = Writer::new(tensors).to_bytes().unwrap();
    let reader = Reader::from_bytes(packed).unwrap();
    let read = reader.get_tensors().unwrap();

    for (i, (dtype, elem)) in cases.iter().enumerate() {
        let t = read.iter().find(|t| t.name == format!("t{i}")).unwrap();
        assert_eq!(t.dtype, *dtype, "dtype preserved for t{i}");
        assert_eq!(t.shape.to_vec(), vec![n]);
        assert_eq!(t.byte_len(), n * elem, "byte_len preserved for t{i}");
        let materialized: &[u8] = &t.bytes;
        assert_eq!(materialized, &vec![i as u8 + 1; n * elem][..]);
    }
}

#[test]
fn in_memory_round_trip() {
    let packed = Writer::new(vec![f32_tensor("w", &[1.5, 2.5, 3.5], &[3], None)])
        .to_bytes()
        .unwrap();

    let reader = Reader::from_bytes(packed).unwrap();
    let tensors = reader.get_tensors().unwrap();
    assert_eq!(read_f32(&tensors[0]), vec![1.5, 2.5, 3.5]);
}

// Reading from a file backs each tensor with `Bytes::from_file`, read lazily on access.
#[cfg(feature = "std")]
#[test]
fn file_backed_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zc.bpk");
    Writer::new(vec![f32_tensor("w", &[1.5, 2.5, 3.5], &[3], None)])
        .write_to_file(&path)
        .unwrap();

    let reader = Reader::from_file(&path).unwrap();
    let tensors = reader.get_tensors().unwrap();
    assert_eq!(read_f32(&tensors[0]), vec![1.5, 2.5, 3.5]);
}

#[test]
fn read_single_tensor_data_by_name() {
    let packed = Writer::new(vec![
        f32_tensor("a", &[1.0, 2.0], &[2], None),
        f32_tensor("b", &[3.0], &[1], None),
    ])
    .to_bytes()
    .unwrap();

    let reader = Reader::from_bytes(packed).unwrap();
    let raw = reader.tensor_data("a").unwrap();
    let values: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(values, vec![1.0, 2.0]);

    assert!(reader.tensor_data("missing").is_err());
}

#[cfg(feature = "std")]
#[test]
fn file_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.bpk");

    Writer::new(vec![f32_tensor(
        "weight",
        &[1.0, 2.0, 3.0, 4.0],
        &[2, 2],
        Some(1),
    )])
    .with_metadata("producer", "burn-pack")
    .write_to_file(&path)
    .unwrap();

    let reader = Reader::from_file(&path).unwrap();
    let tensors = reader.get_tensors().unwrap();
    assert_eq!(tensors.len(), 1);
    assert_eq!(read_f32(&tensors[0]), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(tensors[0].param_id, Some(1));
    assert_eq!(reader.metadata()["producer"], "burn-pack");
}
