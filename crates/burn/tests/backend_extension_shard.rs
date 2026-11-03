//! Runs a backend extension on a sharded compute device from the client and handlers
//! `ShardedCompute` beside `Fusion` generates, with the handlers given to its Flex members. Every call is compared with
//! the same call on one Flex device.
#![cfg(all(feature = "shard", feature = "fusion", feature = "flex"))]

use burn::backend::{
    Backend, CustomOpRegistry, Dispatch, Flex, backend_extension, ops::FloatTensorOps,
    tensor::FloatTensor,
};
use burn::tensor::{Device, Sharding, Tensor};

#[backend_extension(Flex, Fusion, ShardedCompute)]
pub trait CenterBackend: Backend {
    /// Each row of `x` minus the row's sum, times `scale`.
    #[fusion(dtype = x, shape = x)]
    fn center(x: FloatTensor<Self>, scale: f32) -> FloatTensor<Self>;
}

impl CenterBackend for Flex {
    fn center(x: FloatTensor<Self>, scale: f32) -> FloatTensor<Self> {
        let sums = Flex::float_sum_dim(x.clone(), 1);
        Flex::float_mul_scalar(Flex::float_sub(x, sums), scale.into())
    }
}

fn sharded(registry: CustomOpRegistry<Flex>) -> Device {
    Device::shard_options(&[Device::flex(), Device::flex()])
        .with_custom_ops(registry)
        .init()
        .unwrap()
}

fn center_sharded() -> Device {
    let mut registry = CustomOpRegistry::<Flex>::new();
    CenterBackendHandlers::register(&mut registry);
    sharded(registry)
}

fn input(device: &Device) -> Tensor<2> {
    Tensor::from_data([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 9.0]], device)
}

fn center(x: Tensor<2>) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as CenterBackend>::center(x.into_dispatch(), 2.0))
}

/// A row sum over a column shard would differ from the whole row's, so the op only matches one
/// device when the sharded compute device gathers its input first.
#[test]
fn a_custom_op_on_a_sharded_device_gathers_a_split_input() {
    let device = center_sharded();
    let x = input(&device).with_sharding(Sharding::Split { dim: 1 });

    let out = center(x);

    assert_eq!(out.sharding(), Some(Sharding::Replicated));
    out.into_data()
        .assert_eq(&center(input(&Device::flex())).into_data(), true);
}

#[cfg(feature = "cpu")]
#[test]
fn a_sharded_device_refuses_another_backends_custom_ops() {
    use burn::backend::Cube;
    use burn::tensor::shard::ShardError;

    let refused = Device::shard_options(&[Device::flex(), Device::flex()])
        .with_custom_ops(CustomOpRegistry::<Cube>::new())
        .init();

    assert!(
        matches!(refused, Err(ShardError::CustomOpsBackend { .. })),
        "{refused:?}"
    );
}
