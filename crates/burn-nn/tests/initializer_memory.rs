use burn_core::tensor::{Device, Tensor};
use burn_nn::Initializer;

// Run in isolation with a reporting backend:
// cargo test -p burn-nn --features burn-core/cpu --test initializer_memory -- --ignored
#[test]
#[ignore = "requires a CubeCL backend with memory pool reporting"]
fn orthogonal_initialization_uses_dynamic_pools_for_qr() {
    let device = Device::default();
    device.memory_cleanup();
    let before = device
        .memory_pool_usage()
        .expect("the selected backend must report memory pool usage");

    let shape = [32, 16];
    let parameter = Initializer::Orthogonal { gain: 1.0 }.init::<2, _>(shape, &device);
    let weights: Tensor<2> = parameter.val();
    device.sync().unwrap();
    let initialized = device.memory_pool_usage().unwrap();

    let pools = device
        .memory_pool_report()
        .expect("the selected backend must report dynamic memory pools");
    let weight_bytes = (shape.iter().product::<usize>() * weights.dtype().size()) as u64;
    println!("Before initialization: {before:?}");
    println!("After initialization: {initialized:?}");
    println!("Dynamic pools: {pools:?}");
    device.memory_cleanup();
    let after = device.memory_pool_usage().unwrap();
    println!("After cleanup: {after:?}");
    assert!(
        pools.iter().any(|pool| pool.largest_alloc >= weight_bytes),
        "QR temporaries should use dynamic pools: {pools:?}"
    );
    assert_eq!(
        after.bytes_in_use,
        before.bytes_in_use + weight_bytes,
        "only the final weights should remain allocated after cleanup: {after:?}"
    );
}
