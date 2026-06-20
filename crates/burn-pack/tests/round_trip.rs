//! Write → read round-trip coverage for the burnpack format.

mod common;

use burn_pack::{Bytes, DType, Reader, Scalar, Tensor, Writer};
use common::{f32_tensor, raw_tensor, read_f32};

#[test]
fn single_tensor_round_trip() {
    let tensor = f32_tensor("weight", &[1.0, 2.0, 3.0, 4.0], &[2, 2], Some(7));

    let packed = Writer::new(vec![tensor]).into_bytes().unwrap();
    let reader = Reader::from_bytes(packed).unwrap();
    let tensors = reader.into_tensors().unwrap();

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
    .into_bytes()
    .unwrap();

    let reader = Reader::from_bytes(packed).unwrap();
    // Read names before consuming the reader, then consume it for the tensors.
    assert_eq!(reader.tensor_names(), vec!["alpha", "mango", "zebra"]);
    let names: Vec<_> = reader
        .into_tensors()
        .unwrap()
        .iter()
        .map(|t| t.name.clone())
        .collect();

    assert_eq!(names, vec!["alpha", "mango", "zebra"]);
}

#[test]
fn tensors_with_varied_sizes_map_to_correct_names() {
    // On-disk data order = write order (z, a, m), which differs from the name-sorted order the
    // reader returns. The odd sizes force different alignment gaps between tensors, so this
    // exercises the gap/offset arithmetic of the view-based loader: each tensor's zero-copy
    // window must still come back attached to the right name.
    let z: Vec<f32> = (0..5).map(|i| 100.0 + i as f32).collect();
    let a: Vec<f32> = (0..3).map(|i| i as f32).collect();
    let m: Vec<f32> = (0..7).map(|i| 50.0 + i as f32).collect();

    let packed = Writer::new(vec![
        f32_tensor("zebra", &z, &[5], None),
        f32_tensor("alpha", &a, &[3], None),
        f32_tensor("mango", &m, &[7], None),
    ])
    .into_bytes()
    .unwrap();

    let reader = Reader::from_bytes(packed).unwrap();
    let tensors = reader.into_tensors().unwrap();

    let names: Vec<_> = tensors.iter().map(|t| t.name.clone()).collect();
    assert_eq!(names, vec!["alpha", "mango", "zebra"]);
    assert_eq!(read_f32(&tensors[0]), a);
    assert_eq!(read_f32(&tensors[1]), m);
    assert_eq!(read_f32(&tensors[2]), z);
}

#[test]
fn user_metadata_round_trip() {
    let packed = Writer::new(vec![f32_tensor("w", &[1.0], &[1], None)])
        .with_metadata("producer", "burn-pack")
        .with_metadata("format", "burnpack")
        .into_bytes()
        .unwrap();

    let reader = Reader::from_bytes(packed).unwrap();
    assert_eq!(reader.metadata()["producer"], "burn-pack");
    assert_eq!(reader.metadata()["format"], "burnpack");
    // Per-tensor info comes from the tensors themselves.
    let tensors = reader.into_tensors().unwrap();
    assert_eq!(tensors[0].dtype, DType::F32);
    assert_eq!(tensors[0].shape.to_vec(), vec![1]);
}

#[test]
fn param_id_present_and_absent() {
    let packed = Writer::new(vec![
        f32_tensor("with_id", &[1.0], &[1], Some(123)),
        f32_tensor("without_id", &[2.0], &[1], None),
    ])
    .into_bytes()
    .unwrap();

    let reader = Reader::from_bytes(packed).unwrap();
    let tensors = reader.into_tensors().unwrap();
    let with = tensors.iter().find(|t| t.name == "with_id").unwrap();
    let without = tensors.iter().find(|t| t.name == "without_id").unwrap();

    assert_eq!(with.param_id, Some(123));
    assert_eq!(without.param_id, None);
}

#[test]
fn empty_pack() {
    let packed = Writer::new(vec![]).into_bytes().unwrap();
    let reader = Reader::from_bytes(packed).unwrap();
    assert!(reader.tensor_names().is_empty());
    assert!(reader.into_tensors().unwrap().is_empty());
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

    let packed = Writer::new(tensors).into_bytes().unwrap();
    let reader = Reader::from_bytes(packed).unwrap();
    let read = reader.into_tensors().unwrap();

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
        .into_bytes()
        .unwrap();

    let reader = Reader::from_bytes(packed).unwrap();
    let tensors = reader.into_tensors().unwrap();
    assert_eq!(read_f32(&tensors[0]), vec![1.5, 2.5, 3.5]);
}

// Reading from a file backs each tensor with a file-backed `Bytes::view` window, read lazily on
// access.
#[cfg(feature = "std")]
#[test]
fn file_backed_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zc.bpk");
    Writer::new(vec![f32_tensor("w", &[1.5, 2.5, 3.5], &[3], None)])
        .write_to_file(&path)
        .unwrap();

    let reader = Reader::from_file(&path).unwrap();
    let tensors = reader.into_tensors().unwrap();
    assert_eq!(read_f32(&tensors[0]), vec![1.5, 2.5, 3.5]);
}

// A `shared()` tensor exposes zero-copy `view` windows, so the writer streams it
// in bounded chunks rather than materializing it whole. Data larger than the
// writer's internal chunk size exercises the multi-chunk path (several full
// windows plus a partial tail); the concatenated windows must reproduce the
// original bytes exactly.
#[test]
fn shared_tensor_chunked_write_round_trip() {
    // Comfortably larger than the writer's 8 MiB chunk size, and not a whole
    // multiple of it, so the final chunk is partial.
    let len = 8 * 1024 * 1024 + 4096;
    let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();

    let bytes = Bytes::from_bytes_vec(data.clone()).shared();
    let tensor = Tensor::new("big".to_string(), DType::U8, vec![len], None, bytes);

    let packed = Writer::new(vec![tensor]).into_bytes().unwrap();
    let reader = Reader::from_bytes(packed).unwrap();
    let read = reader.into_tensors().unwrap();

    assert_eq!(read.len(), 1);
    let materialized: &[u8] = &read[0].bytes;
    assert_eq!(materialized.len(), len);
    assert_eq!(materialized, &data[..]);
}

#[test]
fn read_single_tensor_data_by_name() {
    let packed = Writer::new(vec![
        f32_tensor("a", &[1.0, 2.0], &[2], None),
        f32_tensor("b", &[3.0], &[1], None),
    ])
    .into_bytes()
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
    assert_eq!(reader.metadata()["producer"], "burn-pack");
    let tensors = reader.into_tensors().unwrap();
    assert_eq!(tensors.len(), 1);
    assert_eq!(read_f32(&tensors[0]), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(tensors[0].param_id, Some(1));
}

#[test]
fn extensionless_path_appends_bpk() {
    let dir = tempfile::tempdir().unwrap();
    // No extension on the path: the writer should append `.bpk`, and the reader should find it
    // when given the same extension-less path.
    let path = dir.path().join("model");

    Writer::new(vec![f32_tensor("weight", &[1.0, 2.0], &[2], Some(7))])
        .write_to_file(&path)
        .unwrap();

    assert!(
        path.with_extension("bpk").exists(),
        "writer should have created `model.bpk`"
    );
    assert!(
        !path.exists(),
        "no extension-less `model` file should exist"
    );

    let reader = Reader::from_file(&path).unwrap();
    let tensors = reader.into_tensors().unwrap();
    assert_eq!(read_f32(&tensors[0]), vec![1.0, 2.0]);
    assert_eq!(tensors[0].param_id, Some(7));
}

#[test]
fn typed_scalars_round_trip() {
    let packed = Writer::new(vec![f32_tensor("w", &[1.0], &[1], None)])
        .with_scalar("step", Scalar::UInt(42))
        .with_scalar("lr", Scalar::Float(0.001))
        .with_scalar("flag", Scalar::Bool(true))
        .with_scalar("offset", Scalar::Int(-3))
        .into_bytes()
        .unwrap();

    let reader = Reader::from_bytes(packed).unwrap();
    let scalars = reader.scalars();
    assert_eq!(scalars.get("step"), Some(&Scalar::UInt(42)));
    assert_eq!(scalars.get("lr"), Some(&Scalar::Float(0.001)));
    assert_eq!(scalars.get("flag"), Some(&Scalar::Bool(true)));
    assert_eq!(scalars.get("offset"), Some(&Scalar::Int(-3)));
}

#[test]
fn scalars_absent_by_default() {
    // A pack written without scalars exposes an empty scalar map (backward compatible: files
    // predating scalar support simply omit the field and default to empty on read).
    let packed = Writer::new(vec![f32_tensor("w", &[1.0], &[1], None)])
        .into_bytes()
        .unwrap();
    let reader = Reader::from_bytes(packed).unwrap();
    assert!(reader.scalars().is_empty());
}
