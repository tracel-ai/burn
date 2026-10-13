use burn_tensor::{
    Device, Distribution, GroupPlacement,
    GroupPlacement::{Partial, Replicated, Sharded},
    Tensor, TensorData, Tolerance, activation, module,
};

#[test]
fn megatron_mlp_on_a_device_group_matches_one_device() {
    let expected = mlp(&Device::flex(), [Replicated; 5]).output.into_data();
    for members in [1, 2, 3, 4] {
        let split = mlp(
            &group(members),
            [
                Replicated,
                Sharded { dim: 1 },
                Sharded { dim: 0 },
                Sharded { dim: 0 },
                Replicated,
            ],
        );

        assert_eq!(
            split.hidden.group_placement(),
            Some(Sharded { dim: 1 }),
            "{members} members"
        );
        split
            .output
            .into_data()
            .assert_approx_eq::<f32>(&expected, tolerance());
    }
}

#[test]
fn a_tensor_moved_onto_a_group_is_replicated_and_moves_back_whole() {
    let data = values([4, 6], 1);
    let tensor = Tensor::<2>::from_data(data.clone(), &Device::flex()).to_device(&group(2));
    assert_eq!(tensor.group_placement(), Some(Replicated));

    let back = tensor
        .place_in_group(Sharded { dim: 1 })
        .to_device(&Device::flex());
    assert_eq!(back.group_placement(), None);
    back.into_data().assert_approx_eq::<f32>(&data, tolerance());
}

#[test]
fn a_tensor_placed_mid_graph_passes_its_gradient_through() {
    let device = group(2).autodiff();
    let x = Tensor::<1>::from_data(values([4], 1), &device).require_grad();
    let y = x.clone().mul_scalar(2.0).place_in_group(Sharded { dim: 0 }) + x.clone();
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
        .place_in_group(Sharded { dim: 1 })
        .to_device(&Device::flex());
    from_every_member
        .into_data()
        .assert_approx_eq::<f32>(&expected, tolerance());
}

#[test]
#[should_panic(expected = "The sharded dim must be smaller than the tensor dimension")]
fn a_split_along_a_dim_the_tensor_lacks_is_refused() {
    Tensor::<2>::zeros([4, 6], &group(2)).place_in_group(Sharded { dim: 2 });
}

#[test]
#[should_panic(expected = "Only a partial sum can be placed as a partial sum")]
fn a_whole_tensor_is_not_placed_as_a_partial_sum() {
    Tensor::<2>::zeros([4, 6], &group(2)).place_in_group(Partial);
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
    let tensor = Tensor::<1>::from_data(data.clone(), &first).place_in_group(Sharded { dim: 0 });
    let moved = tensor.clone().to_device(&second);
    assert_eq!(tensor.group_placement(), Some(Sharded { dim: 0 }));
    assert_eq!(moved.group_placement(), Some(Replicated));
    moved
        .into_data()
        .assert_approx_eq::<f32>(&data, tolerance());
}

struct Mlp {
    hidden: Tensor<2>,
    output: Tensor<2>,
}

/// Placements of x, w1, b1, w2 and b2.
fn mlp(device: &Device, [px, pw1, pb1, pw2, pb2]: [GroupPlacement; 5]) -> Mlp {
    let x = placed::<2>(values([4, 6], 1), device, px);
    let w1 = placed::<2>(values([6, 12], 2), device, pw1);
    let b1 = placed::<1>(values([12], 3), device, pb1);
    let w2 = placed::<2>(values([12, 5], 4), device, pw2);
    let b2 = placed::<1>(values([5], 5), device, pb2);

    let hidden = activation::relu(module::linear(x, w1, Some(b1)));
    let output = module::linear(hidden.clone(), w2, Some(b2));
    Mlp { hidden, output }
}

fn group(members: usize) -> Device {
    Device::group(&vec![Device::flex(); members])
}

fn placed<const D: usize>(
    data: TensorData,
    device: &Device,
    placement: GroupPlacement,
) -> Tensor<D> {
    let tensor = Tensor::from_data(data, device);
    match tensor.group_placement() {
        Some(_) => tensor.place_in_group(placement),
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
