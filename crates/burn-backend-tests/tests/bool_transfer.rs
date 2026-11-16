//! Boolean storage across CubeCL devices with different dtype support.
//!
//! Run with `--no-default-features --features std,cpu,wgpu --test bool_transfer`,
//! and again with `fusion` enabled. The wgpu validation and CI suites run both configurations.

// Without a native shader compiler, wgpu uses WGSL even on a Vulkan adapter.
#![cfg(not(any(feature = "vulkan", feature = "wgpu-msl")))]

use burn_tensor::{Bool, BoolStore, Bytes, DType, Device, DeviceConfig, Tensor, TensorData};

#[test]
fn bool_move_preserves_supported_storage_despite_destination_default() {
    let cpu = Device::cpu();
    let wgpu = Device::wgpu(Default::default());
    let dtype = DType::Bool(BoolStore::U32);
    let data = TensorData::from_bytes(Bytes::from_elems(vec![2u32, 0, 255]), [3], dtype);
    assert!(cpu.supports_dtype(dtype));
    assert_eq!(cpu.settings().bool_dtype, BoolStore::U8);
    let tensor = Tensor::<1, Bool>::from_data(data.clone(), &wgpu);

    let moved = tensor.to_device(&cpu);

    assert_eq!(moved.dtype(), dtype);
    moved.into_data().assert_eq(&data, true);
}

#[test]
fn bool_move_from_cpu_preserves_strided_nonzero_values() {
    let cpu = Device::cpu();
    let wgpu = Device::wgpu(Default::default());
    let u8_bool = DType::Bool(BoolStore::U8);
    let u32_bool = DType::Bool(BoolStore::U32);
    assert!(!wgpu.supports_dtype(u8_bool));

    let data = TensorData::from_bytes_vec(vec![2, 0, 255, 0, 1, 2], [2, 3], u8_bool);
    let tensor = Tensor::<2, Bool>::from_data(data, &cpu).swap_dims(0, 1);
    assert_eq!(tensor.dtype(), u8_bool);

    let moved = tensor.to_device(&wgpu);

    assert_eq!(moved.dtype(), u32_bool);
    moved.bool_not().into_data().assert_eq(
        &TensorData::from([[false, true], [true, false], [false, false]]).convert_dtype(u32_bool),
        true,
    );
}

#[test]
fn bool_move_to_another_backend_uses_configured_fallback_for_unsupported_storage() {
    let wgpu = Device::wgpu(Default::default());
    let mut flex = Device::flex();
    flex.configure(DeviceConfig::default().bool_dtype(BoolStore::U8))
        .unwrap();
    assert!(!flex.supports_dtype(DType::Bool(BoolStore::U32)));
    let data = TensorData::from_bytes(
        Bytes::from_elems(vec![2u32, 0, 255]),
        [3],
        DType::Bool(BoolStore::U32),
    );

    let moved = Tensor::<1, Bool>::from_data(data, &wgpu).to_device(&flex);

    let dtype = DType::Bool(BoolStore::U8);
    assert_eq!(moved.dtype(), dtype);
    moved.bool_not().into_data().assert_eq(
        &TensorData::from([false, true, false]).convert_dtype(dtype),
        true,
    );
}

#[test]
fn bool_move_can_convert_to_explicitly_requested_storage_on_destination() {
    use burn_backend::{TensorMetadata, ops::BoolTensorOps};
    use burn_dispatch::Dispatch;

    let cpu = Device::cpu();
    let wgpu = Device::wgpu(Default::default());
    let data = TensorData::from_bytes(
        Bytes::from_elems(vec![2u32, 0, 255, 1]),
        [4],
        DType::Bool(BoolStore::U32),
    );
    let tensor = Dispatch::bool_from_data(data, wgpu.as_dispatch());
    // WGSL cannot write U8, so an explicit U8 request must convert on the CPU.
    let moved = Dispatch::bool_to_device(tensor, cpu.as_dispatch(), BoolStore::U8);

    let dtype = DType::Bool(BoolStore::U8);
    assert_eq!(moved.dtype(), dtype);
    let moved = burn_backend::read_sync(Dispatch::bool_into_data(moved)).unwrap();
    assert_eq!(moved.as_slice::<u8>().unwrap(), &[1, 0, 1, 1]);
}

#[test]
fn empty_bool_move_adapts_unsupported_storage_and_preserves_supported_storage() {
    let cpu = Device::cpu();
    let wgpu = Device::wgpu(Default::default());
    let tensor = Tensor::<2, Bool>::empty([0, 3], &cpu);

    let moved = tensor.to_device(&wgpu);
    let dtype = DType::Bool(BoolStore::U32);
    let expected = TensorData::from_bytes_vec(vec![], [0, 3], dtype);
    assert_eq!(moved.dtype(), dtype);
    moved.to_data().assert_eq(&expected, true);
    let returned = moved.to_device(&cpu);
    assert_eq!(returned.dtype(), dtype);
    returned.into_data().assert_eq(&expected, true);
}
