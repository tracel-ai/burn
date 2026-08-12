//! Driving a [`burn_pack::Writer`] from snapshots directly, without a `Module` or a
//! `BurnpackStore` in between.
//!
//! This is the path a codegen tool takes when it holds weights read out of some other format
//! and wants a `.bpk` out of them. Being an integration test, it also pins the surface such a
//! crate actually sees: the `burn_pack` re-export, the `TensorEntry` impl on `TensorSnapshot`,
//! and the `From<Tensor>` conversion back.

#![cfg(feature = "burnpack")]

use burn_core::module::ParamId;
use burn_core::tensor::Tensor;
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
