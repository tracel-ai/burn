//! How an operation's output dtype is resolved from its inputs': the dtype of the inputs that
//! propagate, the widest on a tie, a module parameter never deciding.

use super::*;
use burn_tensor::{DType, DTypePropagation, FloatDType, TensorData, module};

fn signal(dtype: FloatDType) -> TestTensor<2> {
    TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &Default::default()).cast(dtype)
}

fn weight(dtype: FloatDType) -> TestTensor<2> {
    signal(dtype).with_propagation(DTypePropagation::Inhibit)
}

#[test]
fn a_signal_survives_a_wider_weight() {
    let out = signal(FloatDType::F16).matmul(weight(FloatDType::F32));
    assert_eq!(out.dtype(), DType::F16);
    assert_eq!(out.propagation(), DTypePropagation::Propagate);
}

#[test]
fn a_signal_survives_a_wider_bias() {
    let out = weight(FloatDType::F32) + signal(FloatDType::BF16);
    assert_eq!(out.dtype(), DType::BF16);
}

#[test]
fn the_wider_of_two_signals_wins() {
    let out = signal(FloatDType::F32) + signal(FloatDType::F16);
    assert_eq!(out.dtype(), DType::F32);
}

#[test]
fn signals_as_wide_promote_to_f32() {
    let out = signal(FloatDType::F16) * signal(FloatDType::BF16);
    assert_eq!(out.dtype(), DType::F32);
}

#[test]
fn weights_alone_inhibit() {
    let out = weight(FloatDType::F32).transpose() + weight(FloatDType::F16);
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.propagation(), DTypePropagation::Inhibit);
}

#[test]
fn a_cast_propagates() {
    let out = weight(FloatDType::F32).cast(FloatDType::F16);
    assert_eq!(out.propagation(), DTypePropagation::Propagate);
}

#[test]
fn an_embedding_lookup_inhibits_like_its_table() {
    let table = weight(FloatDType::F32);
    let indices = TestTensorInt::<2>::from_data([[1, 0]], &table.device());
    let out = module::embedding(table, indices);
    assert_eq!(out.propagation(), DTypePropagation::Inhibit);
}

#[test]
fn a_write_keeps_its_target_type() {
    let target = weight(FloatDType::F32);
    let value = TestTensor::<2>::from_data([[9.0, 9.0]], &target.device()).cast(FloatDType::F16);
    let out = target.slice_assign([0..1, 0..2], value);
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.propagation(), DTypePropagation::Inhibit);
    out.into_data()
        .assert_eq(&TensorData::from([[9.0f32, 9.0], [3.0, 4.0]]), false);
}

#[cfg(feature = "quantization")]
mod quantized {
    use super::*;
    use burn_tensor::{
        Tolerance,
        quantization::{QuantScheme, QuantStore, QuantValue},
    };

    fn int8() -> QuantScheme {
        QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native)
    }

    #[test]
    fn a_quantized_signal_survives_a_float_weight() {
        let x = signal(FloatDType::F32).quantize_dynamic(&int8());
        let out = x.matmul(weight(FloatDType::F32));
        assert_eq!(out.dtype(), DType::QFloat(int8()));
        // The product of the decoded signal, rounded once more into the signal's format.
        let expected = signal(FloatDType::F32)
            .quantize_dynamic(&int8())
            .dequantize()
            .matmul(weight(FloatDType::F32))
            .quantize_dynamic(&int8())
            .dequantize()
            .into_data();
        out.dequantize()
            .into_data()
            .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
    }

    #[test]
    fn a_quantized_weight_never_lends_its_format() {
        let w = weight(FloatDType::F32)
            .quantize_dynamic(&int8())
            .with_propagation(DTypePropagation::Inhibit);
        let out = signal(FloatDType::F32).matmul(w);
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.propagation(), DTypePropagation::Propagate);
    }

    #[test]
    fn a_quantized_signal_reaches_through_unary_operations() {
        let x = signal(FloatDType::F32).quantize_dynamic(&int8());
        let out = x.exp().reshape([4]).sum();
        assert_eq!(out.dtype(), DType::QFloat(int8()));
    }

    #[test]
    fn a_dequantized_branch_keeps_a_float_residual() {
        let residual = signal(FloatDType::F32);
        let branch = signal(FloatDType::F32).quantize_dynamic(&int8());
        let out = residual + branch.dequantize();
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn a_quantized_table_selects_its_rows_as_stored() {
        let table = weight(FloatDType::F32)
            .quantize_dynamic(&int8())
            .with_propagation(DTypePropagation::Inhibit);
        let indices = TestTensorInt::<2>::from_data([[1, 0]], &table.device());
        let out = module::embedding(table.clone(), indices);
        assert_eq!(out.dtype(), table.dtype());
        out.dequantize().into_data().assert_approx_eq::<FloatElem>(
            &table.dequantize().flip([0]).reshape([1, 2, 2]).into_data(),
            Tolerance::default(),
        );
    }
}
