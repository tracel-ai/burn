mod dummy;

use std::sync::Arc;

use crate::dummy::{client, DummyDevice, DummyElementwiseAddition};

use serial_test::serial;

#[test]
fn created_resource_is_the_same_when_read() {
    let client = client(&DummyDevice);
    let resource = Vec::from([0, 1, 2]);
    let resource_description = client.create(&resource);

    let obtained_resource = client.read(&resource_description);

    assert_eq!(resource, obtained_resource.read())
}

#[test]
fn empty_allocates_memory() {
    let client = client(&DummyDevice);
    let size = 4;
    let resource_description = client.empty(size);
    let empty_resource = client.read(&resource_description);

    assert_eq!(empty_resource.read().len(), 4);
}

#[test]
fn execute_elementwise_addition() {
    let client = client(&DummyDevice);
    let lhs = client.create(&[0, 1, 2]);
    let rhs = client.create(&[4, 4, 4]);
    let out = client.empty(3);

    client.execute(Arc::new(DummyElementwiseAddition), &[&lhs, &rhs, &out]);

    let obtained_resource = client.read(&out);

    assert_eq!(obtained_resource.read(), Vec::from([4, 5, 6]))
}

#[test]
#[serial]
#[cfg(feature = "std")]
fn autotune_basic_addition_execution() {
    let client = client(&DummyDevice);

    let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
    let lhs = client.create(&[0, 1, 2]);
    let rhs = client.create(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = &[&lhs, &rhs, &out];

    let addition_autotune_kernel = dummy::AdditionAutotuneKernel::new(shapes);
    client.execute_autotune(Box::new(addition_autotune_kernel), handles);

    let obtained_resource = client.read(&out);

    // If slow kernel was selected it would output [0, 1, 2]
    assert_eq!(obtained_resource.read(), Vec::from([4, 5, 6]));
}

#[test]
#[serial]
#[cfg(feature = "std")]
fn autotune_basic_multiplication_execution() {
    let client = client(&DummyDevice);

    let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
    let lhs = client.create(&[0, 1, 2]);
    let rhs = client.create(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = &[&lhs, &rhs, &out];

    let multiplication_autotune_kernel = dummy::MultiplicationAutotuneKernel::new(shapes);
    client.execute_autotune(Box::new(multiplication_autotune_kernel), handles);

    let obtained_resource = client.read(&out);

    // If slow kernel was selected it would output [0, 1, 2]
    assert_eq!(obtained_resource.read(), Vec::from([0, 4, 8]));
}

#[test]
#[serial]
#[cfg(feature = "std")]
fn autotune_cache_hit_test() {
    let client = client(&DummyDevice);

    let shapes_1 = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
    let lhs_1 = client.create(&[0, 1, 2]);
    let rhs_1 = client.create(&[4, 4, 4]);
    let out_1 = client.empty(3);
    let handles_1 = &[&lhs_1, &rhs_1, &out_1];

    let shapes_2 = vec![vec![1, 4], vec![1, 4], vec![1, 4]];
    let lhs_2 = client.create(&[0, 1, 2, 3]);
    let rhs_2 = client.create(&[5, 6, 7, 8]);
    let out_2 = client.empty(4);
    let handles_2 = &[&lhs_2, &rhs_2, &out_2];

    let cache_test_autotune_kernel_1 = dummy::CacheTestAutotuneKernel::new(shapes_1);
    let cache_test_autotune_kernel_2 = dummy::CacheTestAutotuneKernel::new(shapes_2);
    client.execute_autotune(Box::new(cache_test_autotune_kernel_1), handles_1);
    client.execute_autotune(Box::new(cache_test_autotune_kernel_2), handles_2);

    let obtained_resource = client.read(&out_2);

    // Cache should be hit, so CacheTestFastOn3 should be used, returning lhs
    assert_eq!(obtained_resource.read(), Vec::from([0, 1, 2, 3]));
}

#[test]
#[serial]
#[cfg(feature = "std")]
fn autotune_cache_miss_test() {
    let client = client(&DummyDevice);

    let shapes_1 = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
    let lhs_1 = client.create(&[0, 1, 2]);
    let rhs_1 = client.create(&[4, 4, 4]);
    let out_1 = client.empty(3);
    let handles_1 = &[&lhs_1, &rhs_1, &out_1];

    let shapes_2 = vec![vec![1, 5], vec![1, 5], vec![1, 5]];
    let lhs_2 = client.create(&[0, 1, 2, 3, 4]);
    let rhs_2 = client.create(&[5, 6, 7, 8, 9]);
    let out_2 = client.empty(5);
    let handles_2 = &[&lhs_2, &rhs_2, &out_2];

    let cache_test_autotune_kernel_1 = dummy::CacheTestAutotuneKernel::new(shapes_1);
    let cache_test_autotune_kernel_2 = dummy::CacheTestAutotuneKernel::new(shapes_2);
    client.execute_autotune(Box::new(cache_test_autotune_kernel_1), handles_1);
    client.execute_autotune(Box::new(cache_test_autotune_kernel_2), handles_2);

    let obtained_resource = client.read(&out_2);

    // Cache should be missed, so CacheTestSlowOn3 (but faster on 5) should be used, returning rhs
    assert_eq!(obtained_resource.read(), Vec::from([5, 6, 7, 8, 9]));
}

#[test]
#[serial]
#[cfg(feature = "std")]
fn autotune_operation_with_parameters() {
    let client = client(&DummyDevice);

    let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
    let lhs = client.create(&[0, 1, 2]);
    let rhs = client.create(&[4, 4, 4]);
    let out = client.empty(3);
    let info = client.create(&[9]);
    let handles = &[&lhs, &rhs, &out, &info];

    let addition_autotune_kernel = dummy::ParameterTestAutotuneKernel::new(shapes, info.clone());
    client.execute_autotune(Box::new(addition_autotune_kernel), handles);

    let obtained_resource = client.read(&out);

    assert_eq!(obtained_resource.read(), Vec::from([13, 14, 15]));
}
