//! Driving a [`burn_pack::Writer`] from collected tensors directly, without a `Module` or a
//! `BurnpackStore` in between.
//!
//! This is the path a codegen tool takes when it holds weights read out of some other format
//! and wants a `.bpk` out of them. Being an integration test, it also pins the surface such a
//! crate actually sees: the `burn_pack` re-export, and `bridge`'s constructors and accessors,
//! which are what let it hand a `Writer` tensors it never has to materialize itself.

// Needs `std`: the file round trip below goes through burn-pack's `write_to_file` /
// `from_file`, which that crate gates on its own `std` feature.
#![cfg(feature = "std")]

use burn_core::tensor::quantization::{QuantScheme, QuantValue};
use burn_core::tensor::{Distribution, Tensor, shape};
use burn_store::bridge;
use burn_store::burn_pack::{Reader, Tensor as PackTensor, Writer};

fn tensor(name: &str, values: [[f32; 2]; 2]) -> PackTensor {
    let device = Default::default();
    let tensor = Tensor::<2>::from_data(values, &device);

    bridge::from_tensor(&tensor, name.to_string(), None)
}

#[test]
fn tensors_write_and_read_back_without_a_module() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.bpk");

    let tensors = vec![
        tensor("encoder.weight", [[1.0, 2.0], [3.0, 4.0]]),
        tensor("decoder.weight", [[5.0, 6.0], [7.0, 8.0]]),
    ];

    // Each tensor is read back from its device only when the writer reaches it.
    Writer::new(tensors)
        .with_metadata("producer", "direct-writer-test")
        .write_to_file(&path)
        .unwrap();

    let reader = Reader::from_file(&path).unwrap();
    assert_eq!(reader.metadata()["producer"], "direct-writer-test");

    let restored = reader.into_tensors().unwrap();

    let paths: Vec<String> = restored.iter().map(|t| t.name.clone()).collect();
    assert_eq!(paths, ["decoder.weight", "encoder.weight"]);

    let decoder = bridge::to_data(&restored[0]).unwrap();
    assert_eq!(
        decoder.try_to_vec::<f32>().unwrap(),
        vec![5.0, 6.0, 7.0, 8.0]
    );
}

/// Quantized is the case where the declared byte length is reconstructed from the scheme
/// rather than observed, so it is where the writer's reserve-then-check plumbing would
/// reject a save. The scheme/shape sweep behind that arithmetic is pinned at the unit level
/// (`bridge.rs`); this drives one tensor that carries both risk axes - a sub-byte value type
/// and a value-byte count that is not a multiple of the scale alignment - through the writer
/// and back. (Both axes hold because the test backend's `quantize_dynamic` stores values
/// natively, one `i8` each; a backend honoring the default `PackedU32` store would produce
/// 4-byte-exact counts and exercise neither.)
#[test]
fn a_quantized_tensor_writes_and_reads_back() {
    let device = Default::default();
    let dims = shape![5, 5];

    let quantized = Tensor::<2>::random(dims.clone(), Distribution::Default, &device)
        .quantize_dynamic(&QuantScheme::default().with_value(QuantValue::Q4S));

    let packed = Writer::new(vec![bridge::from_tensor(
        &quantized,
        "weight".to_string(),
        None,
    )])
    .into_bytes()
    .unwrap();

    let restored = Reader::from_bytes(packed).unwrap().into_tensors().unwrap();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].shape, dims);
}
