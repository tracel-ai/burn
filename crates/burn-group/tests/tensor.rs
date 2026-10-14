use burn_tensor::{
    Device, Distribution, Sharding,
    Sharding::{PartialSum, Replicated, Split},
    Tensor, TensorData, Tolerance, activation, module,
};

#[test]
fn megatron_mlp_on_a_device_group_matches_one_device() {
    check_megatron_mlp([4, 6]);
}

#[test]
fn batched_megatron_mlp_keeps_its_hidden_split_through_the_reshapes_of_linear() {
    check_megatron_mlp([2, 4, 6]);
}

#[test]
fn silu_keeps_a_split() {
    let data = values([4, 6], 1);
    let expected =
        activation::silu(Tensor::<2>::from_data(data.clone(), &Device::flex())).into_data();

    let split = activation::silu(sharded::<2>(data, &group(2), Split { dim: 1 }));

    assert_eq!(split.sharding(), Some(Split { dim: 1 }));
    split
        .into_data()
        .assert_approx_eq::<f32>(&expected, tolerance());
}

#[test]
fn a_tensor_moved_onto_a_group_is_replicated_and_moves_back_whole() {
    let data = values([4, 6], 1);
    let tensor = Tensor::<2>::from_data(data.clone(), &Device::flex()).to_device(&group(2));
    assert_eq!(tensor.sharding(), Some(Replicated));

    let back = tensor
        .with_sharding(Split { dim: 1 })
        .to_device(&Device::flex());
    assert_eq!(back.sharding(), None);
    back.into_data().assert_approx_eq::<f32>(&data, tolerance());
}

#[test]
fn a_tensor_resharded_mid_graph_passes_its_gradient_through() {
    let device = group(2).autodiff();
    let x = Tensor::<1>::from_data(values([4], 1), &device).require_grad();
    let y = x.clone().mul_scalar(2.0).with_sharding(Split { dim: 0 }) + x.clone();
    let grads = y.sum().backward();

    let grad = x.grad(&grads).expect("x has a gradient");
    grad.into_data()
        .assert_approx_eq::<f32>(&TensorData::from([3.0f32; 4]), tolerance());
}

#[test]
fn a_random_tensor_is_the_same_on_every_member() {
    let device = group(2);
    let random = Tensor::<2>::random([4, 6], Distribution::Default, &device);
    let shifted = random.clone() + Tensor::ones([4, 6], &device);

    let expected = random
        .to_device(&Device::flex())
        .add_scalar(1.0)
        .into_data();
    let from_every_member = shifted
        .with_sharding(Split { dim: 1 })
        .to_device(&Device::flex());
    from_every_member
        .into_data()
        .assert_approx_eq::<f32>(&expected, tolerance());
}

#[test]
#[should_panic(expected = "The split dim must be smaller than the tensor dimension")]
fn a_split_along_a_dim_the_tensor_lacks_is_refused() {
    Tensor::<2>::zeros([4, 6], &group(2)).with_sharding(Split { dim: 2 });
}

#[test]
#[should_panic(expected = "Only a partial sum can be sharded as a partial sum")]
fn a_whole_tensor_is_not_sharded_as_a_partial_sum() {
    Tensor::<2>::zeros([4, 6], &group(2)).with_sharding(PartialSum);
}

#[test]
#[should_panic(expected = "A device group needs at least one device")]
fn an_empty_group_is_refused() {
    Device::group(&[]);
}

#[test]
fn groups_of_the_same_devices_are_apart() {
    let (first, second) = (group(2), group(2));
    assert_ne!(first, second);

    let data = values([4], 1);
    let tensor = Tensor::<1>::from_data(data.clone(), &first).with_sharding(Split { dim: 0 });
    let moved = tensor.clone().to_device(&second);
    assert_eq!(tensor.sharding(), Some(Split { dim: 0 }));
    assert_eq!(moved.sharding(), Some(Replicated));
    moved
        .into_data()
        .assert_approx_eq::<f32>(&data, tolerance());
}

fn check_megatron_mlp<const D: usize>(x_shape: [usize; D]) {
    let expected = mlp(&Device::flex(), x_shape, [Replicated; 5])
        .output
        .into_data();
    for num_members in 1..=4 {
        let split = mlp(
            &group(num_members),
            x_shape,
            [
                Replicated,
                Split { dim: 1 },
                Split { dim: 0 },
                Split { dim: 0 },
                Replicated,
            ],
        );

        assert_eq!(
            split.hidden.sharding(),
            Some(Split { dim: D - 1 }),
            "{num_members} members"
        );
        split
            .output
            .into_data()
            .assert_approx_eq::<f32>(&expected, tolerance());
    }
}

struct Mlp<const D: usize> {
    hidden: Tensor<D>,
    output: Tensor<D>,
}

/// Shardings of x, w1, b1, w2 and b2.
fn mlp<const D: usize>(
    device: &Device,
    x_shape: [usize; D],
    [sx, sw1, sb1, sw2, sb2]: [Sharding; 5],
) -> Mlp<D> {
    let x = sharded::<D>(values(x_shape, 1), device, sx);
    let w1 = sharded::<2>(values([6, 12], 2), device, sw1);
    let b1 = sharded::<1>(values([12], 3), device, sb1);
    let w2 = sharded::<2>(values([12, 5], 4), device, sw2);
    let b2 = sharded::<1>(values([5], 5), device, sb2);

    let hidden = activation::relu(module::linear(x, w1, Some(b1)));
    let output = module::linear(hidden.clone(), w2, Some(b2));
    Mlp { hidden, output }
}

fn group(num_members: usize) -> Device {
    Device::group(&vec![Device::flex(); num_members])
}

fn sharded<const D: usize>(data: TensorData, device: &Device, sharding: Sharding) -> Tensor<D> {
    let tensor = Tensor::from_data(data, device);
    match tensor.sharding() {
        Some(_) => tensor.with_sharding(sharding),
        None => tensor,
    }
}

fn values<const D: usize>(shape: [usize; D], seed: usize) -> TensorData {
    let count = shape.iter().product::<usize>();
    let values: Vec<f32> = (0..count)
        .map(|i| ((i * 7 + seed * 13) as f32 * 0.37).sin())
        .collect();
    TensorData::new(values, shape)
}

fn tolerance() -> Tolerance<f32> {
    Tolerance::rel_abs(1e-4, 1e-5)
}
