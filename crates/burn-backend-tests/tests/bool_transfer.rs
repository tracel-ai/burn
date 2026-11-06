//! Boolean storage across CubeCL devices with different dtype support.
//!
//! Run with `--no-default-features --features std,cpu,wgpu --test bool_transfer`,
//! and again with `fusion` enabled. The wgpu validation and CI suites run both configurations.

// Without a native shader compiler, wgpu uses WGSL even on a Vulkan adapter.
#![cfg(not(any(feature = "vulkan", feature = "wgpu-msl")))]

use burn_tensor::{Bool, BoolStore, DType, Device, Tensor, TensorData};

#[test]
fn bool_move_preserves_matching_requested_storage() {
    use burn_backend::{TensorMetadata, ops::BoolTensorOps};
    use burn_dispatch::Dispatch;

    let cpu = Device::cpu();
    let wgpu = Device::wgpu(Default::default());
    let dtype = DType::Bool(BoolStore::U32);
    let data = TensorData::from_bytes_vec(
        [2u32, 0, 255]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect(),
        [3],
        dtype,
    );
    let tensor = Dispatch::bool_from_data(data.clone(), wgpu.as_dispatch());

    // Request U32 even though the native CPU default is U8. A detour through that
    // default would normalize the bytes and reveal an unnecessary conversion.
    let moved = Dispatch::bool_to_device(tensor, cpu.as_dispatch(), BoolStore::U32);

    assert_eq!(moved.dtype(), dtype);
    let moved = burn_backend::read_sync(Dispatch::bool_into_data(moved)).unwrap();
    assert_eq!(
        moved.as_slice::<u32>().unwrap(),
        data.as_slice::<u32>().unwrap()
    );
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
fn bool_move_to_cpu_normalizes_storage_and_nonzero_values() {
    let cpu = Device::cpu();
    let wgpu = Device::wgpu(Default::default());
    let data = TensorData::from_bytes_vec(
        [2u32, 0, 255, 1]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect(),
        [4],
        DType::Bool(BoolStore::U32),
    );
    let moved = Tensor::<1, Bool>::from_data(data, &wgpu).to_device(&cpu);

    let dtype = DType::Bool(BoolStore::U8);
    assert_eq!(moved.dtype(), dtype);
    let local = Tensor::<1, Bool>::from_data([true, true, true, false], &cpu);
    moved.bool_and(local).into_data().assert_eq(
        &TensorData::from([true, false, true, false]).convert_dtype(dtype),
        true,
    );
}

#[test]
fn empty_bool_move_uses_destination_storage_in_both_directions() {
    let cpu = Device::cpu();
    let wgpu = Device::wgpu(Default::default());
    let tensor = Tensor::<2, Bool>::empty([0, 3], &cpu);

    let moved = tensor.to_device(&wgpu);
    let expected = TensorData::new(Vec::<bool>::new(), [0, 3]);

    let dtype = DType::Bool(BoolStore::U32);
    assert_eq!(moved.dtype(), dtype);
    moved
        .to_data()
        .assert_eq(&expected.clone().convert_dtype(dtype), true);
    let returned = moved.to_device(&cpu);
    let dtype = DType::Bool(BoolStore::U8);
    assert_eq!(returned.dtype(), dtype);
    returned
        .into_data()
        .assert_eq(&expected.convert_dtype(dtype), true);
}
