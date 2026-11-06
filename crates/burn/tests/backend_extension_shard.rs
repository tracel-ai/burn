//! Runs a backend extension on a sharded compute device from the client and handlers
//! `ShardedCompute` beside `Fusion` generates, with the handlers given to its Flex members. Every
//! call is compared with the same call on one Flex device.
#![cfg(all(feature = "shard", feature = "fusion", feature = "flex"))]

use burn::backend::{
    Backend, Dispatch, Flex, backend_extension, ir::CustomOpIr, ops::FloatTensorOps,
    tensor::FloatTensor,
};
use burn::tensor::shard::{
    CollectiveMember, CollectiveOpRegistry, CustomOpPlan, CustomOpRegistry, CustomOpRule, OpKind,
    ShardCustomOps,
};
use burn::tensor::{Device, Sharding, Shards, Tensor};

#[backend_extension(Flex, Fusion, ShardedCompute)]
pub trait CenterBackend: Backend {
    /// Each row of `x` minus the row's sum, times `scale`.
    #[fusion(dtype = x, shape = x)]
    fn center(x: FloatTensor<Self>, scale: f32) -> FloatTensor<Self>;

    /// [`center`](Self::center) with each column then scaled by `weight`.
    #[fusion(dtype = x, shape = x)]
    fn center_weighted(x: FloatTensor<Self>, weight: FloatTensor<Self>) -> FloatTensor<Self>;

    /// `x` itself, which a sharded compute device makes whole from every member's rows at once.
    #[fusion(dtype = x, shape = x)]
    fn gather_rows(x: FloatTensor<Self>) -> FloatTensor<Self>;
}

impl CenterBackend for Flex {
    fn center(x: FloatTensor<Self>, scale: f32) -> FloatTensor<Self> {
        let sums = Flex::float_sum_dim(x.clone(), 1);
        Flex::float_mul_scalar(Flex::float_sub(x, sums), scale.into())
    }

    fn center_weighted(x: FloatTensor<Self>, weight: FloatTensor<Self>) -> FloatTensor<Self> {
        let columns = burn::backend::TensorMetadata::shape(&weight)[0];
        let weight = Flex::float_reshape(weight, [1, columns].into());
        Flex::float_mul(Self::center(x, 1.0), weight)
    }

    fn gather_rows(x: FloatTensor<Self>) -> FloatTensor<Self> {
        x
    }
}

/// `center` reads each row whole and every row apart.
struct CenterRule;

impl CustomOpRule for CenterRule {
    fn plan<'a>(&self, _op: &'a CustomOpIr) -> CustomOpPlan<'a> {
        CustomOpPlan::Kind(OpKind::AlongDim { dim: 1 })
    }
}

/// `gather_rows` takes every member's rows and gives each member all of them.
struct GatherRowsRule;

impl CustomOpRule for GatherRowsRule {
    fn plan<'a>(&self, _op: &'a CustomOpIr) -> CustomOpPlan<'a> {
        CustomOpPlan::Collective {
            inputs: vec![Sharding::Split { dim: 0 }],
            outputs: vec![Sharding::Replicated],
        }
    }
}

fn gather_rows_on_every_member(members: &mut [CollectiveMember<'_, Flex>]) {
    let shards = members
        .iter_mut()
        .map(|member| {
            member
                .handles
                .get_float_tensor::<Flex>(&member.op.inputs[0])
        })
        .collect();
    let rows = Flex::float_cat(shards, 0);
    for member in members {
        let out = Flex::float_to_device(rows.clone(), member.device);
        member
            .handles
            .register_float_tensor::<Flex>(&member.op.outputs[0].id, out);
    }
}

fn sharded(custom_ops: impl Into<ShardCustomOps>) -> Device {
    Device::shard_options(&[Device::flex(), Device::flex()])
        .with_custom_ops(custom_ops)
        .init()
        .unwrap()
}

fn handlers() -> CustomOpRegistry<Flex> {
    let mut registry = CustomOpRegistry::<Flex>::new();
    CenterBackendHandlers::register(&mut registry);
    registry
}

fn input(device: &Device) -> Tensor<2> {
    Tensor::from_data([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 9.0]], device)
}

/// [`input`] split along `dim`, each member's chunk made on it.
fn split_input(device: &Device, dim: usize) -> Tensor<2> {
    let members = device
        .shard_members()
        .expect("A sharded compute device has members");
    let shards = input(&Device::flex())
        .chunk(members.len(), dim)
        .into_iter()
        .zip(&members)
        .map(|(chunk, member)| chunk.to_device(member))
        .collect();
    let sharding = Sharding::Split { dim };
    Tensor::from_shards(Shards { shards, sharding }, device).expect("The chunks make up the input")
}

fn center(x: Tensor<2>) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as CenterBackend>::center(x.into_dispatch(), 2.0))
}

fn center_weighted(x: Tensor<2>, weight: Tensor<1>) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as CenterBackend>::center_weighted(
        x.into_dispatch(),
        weight.into_dispatch(),
    ))
}

fn weight(device: &Device) -> Tensor<1> {
    Tensor::from_data([1.0, 2.0, 3.0, 4.0], device)
}

fn gather_rows(x: Tensor<2>) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as CenterBackend>::gather_rows(x.into_dispatch()))
}

/// A row sum over a column shard would differ from the whole row's, so the op only matches one
/// device when the sharded compute device gathers its input first.
#[test]
fn a_custom_op_without_a_rule_gathers_a_split_input() {
    let device = sharded(handlers());
    let x = split_input(&device, 1);

    let out = center(x);

    assert_eq!(out.sharding(), Some(Sharding::Replicated));
    out.into_data()
        .assert_eq(&center(input(&Device::flex())).into_data(), true);
}

#[test]
fn a_custom_op_of_a_kind_runs_on_each_members_rows() {
    let custom_ops = ShardCustomOps::default()
        .with_handlers(handlers())
        .with_rule("CenterBackend::center", CenterRule);
    let device = sharded(custom_ops);
    let x = split_input(&device, 0);

    let out = center(x);

    assert_eq!(out.sharding(), Some(Sharding::Split { dim: 0 }));
    out.into_data()
        .assert_eq(&center(input(&Device::flex())).into_data(), true);
}

/// Without the weight broadcast against `x`, the op would have no rule and gather.
#[test]
fn a_custom_op_along_a_dim_broadcasts_its_further_inputs() {
    let custom_ops = ShardCustomOps::default()
        .with_handlers(handlers())
        .with_rule("CenterBackend::center_weighted", CenterRule);
    let device = sharded(custom_ops);
    let x = split_input(&device, 0);

    let out = center_weighted(x, weight(&device));

    assert_eq!(out.sharding(), Some(Sharding::Split { dim: 0 }));
    let local = Device::flex();
    out.into_data().assert_eq(
        &center_weighted(input(&local), weight(&local)).into_data(),
        true,
    );
}

/// Each member's own handler would return only its rows, so the whole value on every member
/// shows the collective handler ran over all of them, after the replicated input was split for it.
#[test]
fn a_collective_custom_op_runs_once_over_every_member() {
    let mut collectives = CollectiveOpRegistry::<Flex>::new();
    collectives.register("CenterBackend::gather_rows", gather_rows_on_every_member);
    let custom_ops = ShardCustomOps::default()
        .with_handlers(handlers())
        .with_collectives(collectives)
        .with_rule("CenterBackend::gather_rows", GatherRowsRule);
    let device = sharded(custom_ops);
    let x = input(&device);

    let out = gather_rows(x);

    assert_eq!(out.sharding(), Some(Sharding::Replicated));
    out.into_data()
        .assert_eq(&input(&Device::flex()).into_data(), true);
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
