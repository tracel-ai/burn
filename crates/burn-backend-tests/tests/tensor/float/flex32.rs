//! `Flex32` through the matmul, convolution and attention kernels: each op on `Flex32` inputs
//! keeps the dtype and matches the same op on `F32` inputs.

use burn_tensor::module::{
    attention, conv_transpose1d, conv_transpose2d, conv_transpose3d, conv1d, conv2d, conv3d,
    deform_conv2d, linear,
};
use burn_tensor::ops::{ConvOptions, ConvTransposeOptions, DeformConvOptions};
use burn_tensor::{DType, Device, FloatDType, Tensor, TensorData, Tolerance};

/// Deterministic values in `[-1, 1)` of `shape`.
fn values<const D: usize>(shape: [usize; D], seed: usize, device: &Device) -> Tensor<D> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i * 37 + seed * 11) % 97) as f32 / 48.5 - 1.0)
        .collect();
    Tensor::from_data(TensorData::new(data, shape), device)
}

/// Runs `op` on `Flex32` and on `F32` copies of the same inputs; the `Flex32` output keeps its
/// dtype and holds the `F32` values.
fn assert_matches_f32<const D: usize>(op: impl Fn(FloatDType) -> Tensor<D>) {
    let device = Device::default();
    if !device.supports_dtype(DType::Flex32) {
        return;
    }
    let flex = op(FloatDType::Flex32);
    assert_eq!(flex.dtype(), DType::Flex32);
    let expected = op(FloatDType::F32).into_data();
    flex.cast(FloatDType::F32)
        .into_data()
        .assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(1e-3, 1e-3));
}

#[test]
fn flex32_matmul() {
    let device = Device::default();
    // Square, batched with a broadcast lhs, a transposed rhs, and a single row.
    for (lhs, rhs, transpose) in [
        ([1, 2, 2], [1, 2, 2], false),
        ([1, 8, 7], [8, 7, 5], false),
        ([2, 17, 33], [2, 15, 33], true),
        ([3, 1, 64], [3, 64, 16], false),
    ] {
        assert_matches_f32(|dtype| {
            let lhs = values(lhs, 1, &device).cast(dtype);
            let rhs = values(rhs, 2, &device).cast(dtype);
            let rhs = if transpose { rhs.swap_dims(1, 2) } else { rhs };
            lhs.matmul(rhs)
        });
    }
}

/// A linear layer with a fused epilogue, twice: once the plain matmul's autotune has elected a
/// routine, the fused product launches that routine on its own operands.
#[test]
fn flex32_linear_with_epilogue() {
    let device = Device::default();
    for _ in 0..2 {
        assert_matches_f32(|dtype| {
            let x = values([4, 6, 16], 1, &device).cast(dtype);
            let weight = values([16, 8], 2, &device).cast(dtype);
            let bias = values([8], 3, &device).cast(dtype);
            linear(x, weight, Some(bias)).tanh()
        });
    }
}

#[test]
fn flex32_conv() {
    let device = Device::default();
    assert_matches_f32(|dtype| {
        let x = values([2, 3, 9], 1, &device).cast(dtype);
        let weight = values([4, 3, 3], 2, &device).cast(dtype);
        conv1d(x, weight, None, ConvOptions::new([1], [1], [1], 1))
    });
    assert_matches_f32(|dtype| {
        let x = values([2, 3, 7, 6], 1, &device).cast(dtype);
        let weight = values([4, 3, 3, 3], 2, &device).cast(dtype);
        let bias = values([4], 3, &device).cast(dtype);
        conv2d(
            x,
            weight,
            Some(bias),
            ConvOptions::new([1, 2], [1, 0], [1, 1], 1),
        )
    });
    assert_matches_f32(|dtype| {
        let x = values([2, 3, 5, 7], 1, &device).cast(dtype);
        let weight = values([4, 3, 3, 3], 2, &device).cast(dtype);
        conv2d(x, weight, None, ConvOptions::new([1, 1], [1, 1], [1, 1], 1))
    });
    assert_matches_f32(|dtype| {
        let x = values([1, 16, 12, 12], 1, &device).cast(dtype);
        let weight = values([32, 16, 3, 3], 2, &device).cast(dtype);
        conv2d(x, weight, None, ConvOptions::new([1, 1], [1, 1], [1, 1], 1))
    });
    assert_matches_f32(|dtype| {
        let x = values([1, 2, 4, 5, 3], 1, &device).cast(dtype);
        let weight = values([2, 2, 2, 2, 2], 2, &device).cast(dtype);
        conv3d(
            x,
            weight,
            None,
            ConvOptions::new([1, 1, 1], [0, 1, 0], [1, 1, 1], 1),
        )
    });
}

#[test]
fn flex32_conv_transpose() {
    let device = Device::default();
    assert_matches_f32(|dtype| {
        let x = values([2, 3, 5], 1, &device).cast(dtype);
        let weight = values([3, 2, 3], 2, &device).cast(dtype);
        conv_transpose1d(
            x,
            weight,
            None,
            ConvTransposeOptions::new([2], [1], [1], [1], 1),
        )
    });
    assert_matches_f32(|dtype| {
        let x = values([1, 2, 4, 5], 1, &device).cast(dtype);
        let weight = values([2, 3, 3, 3], 2, &device).cast(dtype);
        let bias = values([3], 3, &device).cast(dtype);
        conv_transpose2d(
            x,
            weight,
            Some(bias),
            ConvTransposeOptions::new([2, 1], [1, 1], [1, 0], [1, 1], 1),
        )
    });
    assert_matches_f32(|dtype| {
        let x = values([1, 2, 3, 3, 2], 1, &device).cast(dtype);
        let weight = values([2, 2, 2, 2, 2], 2, &device).cast(dtype);
        conv_transpose3d(
            x,
            weight,
            None,
            ConvTransposeOptions::new([1, 2, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], 1),
        )
    });
}

#[test]
fn flex32_deform_conv2d() {
    let device = Device::default();
    assert_matches_f32(|dtype| {
        let x = values([1, 2, 5, 5], 1, &device).cast(dtype);
        let offset = values([1, 18, 3, 3], 2, &device)
            .cast(dtype)
            .mul_scalar(0.5);
        let weight = values([3, 2, 3, 3], 3, &device).cast(dtype);
        deform_conv2d(
            x,
            offset,
            weight,
            None,
            None,
            DeformConvOptions::new([1, 1], [0, 0], [1, 1], 1, 1),
        )
    });
}

#[test]
fn flex32_attention() {
    let device = Device::default();
    assert_matches_f32(|dtype| {
        let query = values([1, 2, 5, 8], 1, &device).cast(dtype);
        let key = values([1, 2, 6, 8], 2, &device).cast(dtype);
        let value = values([1, 2, 6, 4], 3, &device).cast(dtype);
        attention(query, key, value, None, None, None, Default::default())
    });
}

#[test]
fn flex32_matmul_and_conv2d_gradients() {
    let device = Device::default().autodiff();
    if !device.supports_dtype(DType::Flex32) {
        return;
    }
    // The gradients of each input, with their dtypes.
    let grads = |dtype: FloatDType| {
        let lhs = values([2, 5, 6], 1, &device).cast(dtype).require_grad();
        let rhs = values([2, 6, 3], 2, &device).cast(dtype).require_grad();
        let x = values([1, 2, 5, 5], 3, &device).cast(dtype).require_grad();
        let weight = values([3, 2, 3, 3], 4, &device).cast(dtype).require_grad();
        let product = lhs.clone().matmul(rhs.clone()).sum();
        let conv = conv2d(
            x.clone(),
            weight.clone(),
            None,
            ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
        );
        let grads = (product + conv.sum()).backward();
        let read = |grad: Option<Tensor<3>>| {
            grad.map(|g| (g.dtype(), g.cast(FloatDType::F32).into_data()))
        };
        let read4 = |grad: Option<Tensor<4>>| {
            grad.map(|g| (g.dtype(), g.cast(FloatDType::F32).into_data()))
        };
        [
            read(lhs.grad(&grads)),
            read(rhs.grad(&grads)),
            read4(x.grad(&grads)),
            read4(weight.grad(&grads)),
        ]
        .map(|grad| grad.expect("every input has a gradient"))
    };
    let expected = grads(FloatDType::F32);
    for ((dtype, flex), (_, expected)) in grads(FloatDType::Flex32).into_iter().zip(expected) {
        assert_eq!(dtype, DType::Flex32);
        flex.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(1e-3, 1e-3));
    }
}
