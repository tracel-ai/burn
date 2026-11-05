#![cfg(all(
    feature = "linalg",
    any(
        feature = "cpu",
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan",
        feature = "wgpu"
    )
))]

use burn::{
    nn::Initializer,
    tensor::{Device, Tensor},
};

#[test]
fn orthogonal_initialization_uses_dynamic_pools_for_qr() {
    let device = Device::default();
    device.memory_cleanup();
    let before = device.measure_memory().unwrap().usage.bytes_in_use;

    let weights: Tensor<2> = Initializer::Orthogonal { gain: 1.0 }
        .init([32, 16], &device)
        .val();
    let weight_bytes = (weights.shape().num_elements() * weights.dtype().size()) as u64;
    device.sync().unwrap();

    let pools = device.measure_memory().unwrap().pools;
    assert!(
        pools.iter().any(|pool| pool.largest_alloc >= weight_bytes),
        "QR temporaries should use dynamic pools: {pools:?}"
    );

    device.memory_cleanup();
    assert_eq!(
        device.measure_memory().unwrap().usage.bytes_in_use,
        before + weight_bytes,
        "only the final weights should remain allocated after cleanup"
    );
}
