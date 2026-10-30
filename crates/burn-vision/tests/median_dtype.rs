#![cfg(feature = "flex")]

use burn_core::tensor::{DType, Device, FloatDType, Int, IntDType, Tensor, TensorData};
use burn_vision::median;

// Device defaults can only be configured once, so this needs its own test executable.
#[test]
fn median_preserves_explicit_dtypes_with_different_device_defaults() {
    let mut device = Device::flex();
    device.configure((FloatDType::BF16, IntDType::I8)).unwrap();
    let values = vec![1., 4., 3., 2.]
        .into_iter()
        .map(|v| v / 255.)
        .collect::<Vec<f32>>();
    let images =
        Tensor::<4>::from_data(TensorData::new(values, [1, 1, 2, 2]), (&device, DType::F32));
    let expected = TensorData::new(
        vec![2., 3., 3., 2.]
            .into_iter()
            .map(|v| v / 255.)
            .collect::<Vec<f32>>(),
        [1, 1, 2, 2],
    );

    for size in [3, 7] {
        let output = median(images.clone(), size);
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.into_data(), expected);
    }

    let base = 1_u64 << 63;
    let images = Tensor::<4, Int>::from_data(
        TensorData::new(vec![base + 1, base + 4, base + 3, base + 2], [1, 1, 2, 2]),
        (&device, DType::U64),
    );
    let expected = TensorData::new(vec![base + 2, base + 3, base + 3, base + 2], [1, 1, 2, 2]);
    for size in [3, 7] {
        let output = median(images.clone(), size);
        assert_eq!(output.dtype(), DType::U64);
        assert_eq!(output.into_data(), expected);
    }

    // Gather indices must cover the plane even when the default int is only I8.
    let values = (0..256).map(|i| i as f32).collect::<Vec<_>>();
    let images = Tensor::<4>::from_data(
        TensorData::new(values, [1, 1, 1, 256]),
        (&device.autodiff(), DType::F32),
    )
    .require_grad();
    let output = median(images.clone(), 3);
    assert_eq!(output.clone().into_data(), images.clone().into_data());
    let grads = output.sum().backward();
    assert_eq!(
        images.grad(&grads).unwrap().into_data(),
        TensorData::new(vec![1_f32; 256], [1, 1, 1, 256]),
    );
}
