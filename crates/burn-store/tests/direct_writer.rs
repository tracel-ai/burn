//! Driving a [`burn_pack::Writer`] from snapshots directly, without a `Module` or a
//! `BurnpackStore` in between.
//!
//! This is the path a codegen tool takes when it holds weights read out of some other format
//! and wants a `.bpk` out of them. Being an integration test, it also pins the surface such a
//! crate actually sees: the `burn_pack` re-export, the `TensorEntry` impl on `TensorSnapshot`,
//! and the `From<burn_pack::Tensor>` conversion back.

// Needs `std` as well as `burnpack`: the file round trip below goes through burn-pack's
// `write_to_file` / `from_file`, which that crate gates on its own `std` feature.
#![cfg(all(feature = "burnpack", feature = "std"))]

use burn_core::module::ParamId;
use burn_core::tensor::quantization::{QuantScheme, QuantValue};
use burn_core::tensor::{Distribution, Tensor, shape};
use burn_store::TensorSnapshot;
use burn_store::burn_pack::{Reader, Writer};

fn snapshot(name: &str, values: [[f32; 2]; 2]) -> TensorSnapshot {
    let device = Default::default();
    let tensor = Tensor::<2>::from_data(values, &device);
    TensorSnapshot::from_float(
        &tensor,
        name.split('.').map(str::to_string).collect(),
        vec![],
        ParamId::new(),
    )
}

#[test]
fn snapshots_write_and_read_back_without_a_module() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.bpk");

    let snapshots = vec![
        snapshot("encoder.weight", [[1.0, 2.0], [3.0, 4.0]]),
        snapshot("decoder.weight", [[5.0, 6.0], [7.0, 8.0]]),
    ];

    // `TensorSnapshot: TensorEntry`, so the writer takes them as-is and materializes each one
    // only when it reaches that tensor.
    Writer::new(snapshots)
        .with_metadata("producer", "direct-writer-test")
        .write_to_file(&path)
        .unwrap();

    let reader = Reader::from_file(&path).unwrap();
    assert_eq!(reader.metadata()["producer"], "direct-writer-test");

    let restored: Vec<TensorSnapshot> = reader
        .into_tensors()
        .unwrap()
        .into_iter()
        .map(TensorSnapshot::from)
        .collect();

    let paths: Vec<String> = restored.iter().map(|s| s.full_path()).collect();
    assert_eq!(paths, ["decoder.weight", "encoder.weight"]);

    let decoder = restored[0].to_data().unwrap();
    assert_eq!(decoder.to_vec::<f32>().unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
}

/// Quantized is the case where the declared byte length is reconstructed from the scheme
/// rather than observed, so it is where the writer's reserve-then-check plumbing would
/// reject a save. The scheme/shape sweep behind that arithmetic is pinned at the unit level
/// (`tensor_snapshot.rs`); this drives one tensor that carries both risk axes - a sub-byte
/// value type and a value-byte count that is not a multiple of the scale alignment -
/// through the writer and back.
#[test]
fn quantized_snapshot_writes_and_reads_back() {
    let device = Default::default();
    let dims = shape![5, 5];

    let tensor = Tensor::<2>::random(dims.clone(), Distribution::Default, &device)
        .quantize_dynamic(&QuantScheme::default().with_value(QuantValue::Q4S));
    let snapshot =
        TensorSnapshot::from_float(&tensor, vec!["weight".to_string()], vec![], ParamId::new());

    let packed = Writer::new(vec![snapshot]).into_bytes().unwrap();

    let restored = Reader::from_bytes(packed).unwrap().into_tensors().unwrap();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].shape, dims);
}
