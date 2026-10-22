use burn_tensor::{
    Bool, Device, Distribution, Int, Sharding,
    Sharding::{PartialSum, Replicated, Split},
    ShardingError, Tensor, TensorData, Tolerance, activation,
    group::GroupError,
    module,
    quantization::QuantScheme,
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
fn heads_split_out_of_a_split_hidden_dim_stay_split() {
    let data = values([2, 3, 12], 1);
    let expected = Tensor::<3>::from_data(data.clone(), &Device::flex())
        .reshape([2, 3, 4, 3])
        .into_data();
    for num_members in [2, 4] {
        let heads =
            sharded::<3>(data.clone(), &group(num_members), Split { dim: 2 }).reshape([2, 3, 4, 3]);

        assert_eq!(
            heads.sharding(),
            Some(Split { dim: 2 }),
            "{num_members} members"
        );
        heads
            .into_data()
            .assert_approx_eq::<f32>(&expected, tolerance());
    }
}

#[test]
fn a_split_reshaped_onto_an_empty_dim_adds_a_replicated_tensor() {
    let device = group(2);
    let split = Tensor::<2>::zeros([4, 0], &device).with_sharding(Split { dim: 0 });

    let sum = split.reshape([-1, 4]) + Tensor::zeros([0, 4], &device);

    sum.into_data()
        .assert_approx_eq::<f32>(&TensorData::new(Vec::<f32>::new(), [0, 4]), tolerance());
}

#[test]
fn int_and_bool_tensors_split_and_move_back_whole() {
    let ints = TensorData::from([[1i64, 2, 3, 4], [5, 6, 7, 8]]);
    let bools = TensorData::from([[true, false, true, true], [false, false, true, false]]);

    let split_ints =
        Tensor::<2, Int>::from_data(ints.clone(), &group(2)).with_sharding(Split { dim: 1 });
    let split_bools =
        Tensor::<2, Bool>::from_data(bools.clone(), &group(2)).with_sharding(Split { dim: 0 });

    assert_eq!(split_ints.sharding(), Some(Split { dim: 1 }));
    assert_eq!(split_bools.sharding(), Some(Split { dim: 0 }));
    split_ints.into_data().assert_eq(&ints, false);
    split_bools.into_data().assert_eq(&bools, false);
}

#[test]
fn a_quantized_tensor_on_a_group_reads_back_but_does_not_split() {
    // CUDA packs four quantized values per u32, so the last dim is a multiple of four.
    let data = values([4, 8], 1);
    let scheme = QuantScheme::default();
    let expected = Tensor::<2>::from_data(data.clone(), &Device::flex())
        .quantize_dynamic(&scheme)
        .dequantize()
        .into_data();
    let quantized = Tensor::<2>::from_data(data, &group(2)).quantize_dynamic(&scheme);

    assert!(matches!(
        quantized.try_with_sharding(Split { dim: 0 }),
        Err(ShardingError::Quantized)
    ));
    Tensor::<2>::from_data(quantized.into_data(), &Device::flex())
        .dequantize()
        .into_data()
        .assert_approx_eq::<f32>(&expected, tolerance());
}

#[test]
fn a_quantized_tensor_on_a_group_reports_its_sharding() {
    let quantized = Tensor::<2>::from_data(values([4, 8], 1), &group(2))
        .quantize_dynamic(&QuantScheme::default());

    assert_eq!(quantized.sharding(), Some(Replicated));
}

#[test]
fn a_quantized_replicated_rhs_matmuls_with_a_split_lhs() {
    let device = group(2);
    let x = values([4, 6], 1);
    let weight = Tensor::<2>::from_data(values([6, 8], 2), &device)
        .quantize_dynamic(&QuantScheme::default());
    let expected = Tensor::<2>::from_data(x.clone(), &Device::flex())
        .matmul(Tensor::<2>::from_data(
            weight.clone().into_data(),
            &Device::flex(),
        ))
        .into_data();

    let product = sharded::<2>(x, &device, Split { dim: 1 }).matmul(weight);

    product
        .into_data()
        .assert_approx_eq::<f32>(&expected, tolerance());
}

#[test]
fn a_refused_sharding_leaves_the_tensor_as_it_was() {
    let data = values([2, 6], 1);
    let tensor = Tensor::<2>::from_data(data.clone(), &group(3));

    let refused = tensor.try_with_sharding(Split { dim: 0 });

    assert!(matches!(
        refused,
        Err(ShardingError::DimShorterThanGroup { dim: 0, .. })
    ));
    assert_eq!(tensor.sharding(), Some(Replicated));
    tensor
        .into_data()
        .assert_approx_eq::<f32>(&data, tolerance());
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
fn a_split_along_a_dim_the_tensor_lacks_is_refused() {
    let refused = Tensor::<2>::zeros([4, 6], &group(2)).try_with_sharding(Split { dim: 2 });

    assert!(matches!(
        refused,
        Err(ShardingError::MissingDim { dim: 2, .. })
    ));
}

#[test]
#[should_panic(expected = "is not a partial sum and cannot become one")]
fn a_whole_tensor_is_not_sharded_as_a_partial_sum() {
    Tensor::<2>::zeros([4, 6], &group(2)).with_sharding(PartialSum);
}

#[test]
fn an_empty_group_is_refused() {
    let refused = Device::group_options(&[]).init();

    assert!(matches!(refused, Err(GroupError::NoDevices)));
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
    let members: Vec<Device> = member_devices()
        .into_iter()
        .cycle()
        .take(num_members)
        .collect();
    Device::group_options(&members)
        .init()
        .expect("The test's devices form a group")
}

/// One member per runtime, so no collective between them has a native path.
#[cfg(feature = "test-mixed-runtimes")]
fn member_devices() -> Vec<Device> {
    let cuda = Device::enumerate(burn_tensor::DeviceType::Cuda).into_vec();
    let vulkan = Device::enumerate(burn_tensor::DeviceType::Vulkan).into_vec();
    let members: Vec<Device> = cuda
        .into_iter()
        .take(1)
        .chain(vulkan.into_iter().take(1))
        .collect();
    // Without both, the group spans one runtime and the suite proves nothing it was run for.
    assert_eq!(members.len(), 2, "A CUDA and a Vulkan device are needed");
    assert_ne!(members[0], members[1], "The members are two devices");
    members
}

#[cfg(all(feature = "test-cuda", not(feature = "test-mixed-runtimes")))]
fn member_devices() -> Vec<Device> {
    Device::enumerate(burn_tensor::DeviceType::Cuda).into_vec()
}

#[cfg(not(any(feature = "test-cuda", feature = "test-mixed-runtimes")))]
fn member_devices() -> Vec<Device> {
    vec![Device::flex()]
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

#[cfg(not(any(feature = "test-cuda", feature = "test-mixed-runtimes")))]
fn tolerance() -> Tolerance<f32> {
    Tolerance::rel_abs(1e-4, 1e-5)
}

/// CUDA rounds f32 matmuls through TF32 on cards that have it, about 1% off Flex; a lost or
/// doubled summand is off by far more.
#[cfg(any(feature = "test-cuda", feature = "test-mixed-runtimes"))]
fn tolerance() -> Tolerance<f32> {
    Tolerance::rel_abs(3e-2, 1e-2)
}
