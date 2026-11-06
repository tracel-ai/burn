use burn_tensor::{
    Bool, Device, Distribution, Int, Sharding,
    Sharding::{PartialSum, Replicated, Split},
    Shards, ShardsError, Tensor, TensorData, Tolerance, activation,
    kind::Autodiff,
    module,
    quantization::{QuantScheme, ScaleDtype},
    s,
    shard::{DimSplit, ShardError},
};

use std::sync::{Mutex, PoisonError};

// All random tests share the backend RNG, including those that do not seed it.
static RNG: Mutex<()> = Mutex::new(());

#[test]
fn megatron_mlp_on_a_sharded_compute_device_matches_one_device() {
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

    let split = activation::silu(sharded::<2>(data, &sharded_compute(2), Split { dim: 1 }));

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
        let heads = sharded::<3>(
            data.clone(),
            &sharded_compute(num_members),
            Split { dim: 2 },
        )
        .reshape([2, 3, 4, 3]);

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
fn rotated_heads_stay_split() {
    let (x, angles) = (values([2, 3, 4, 8], 1), values([3, 8], 2));
    let expected = rotate(
        Tensor::from_data(x.clone(), &Device::flex()),
        angles.clone(),
    )
    .into_data();
    for num_members in [2, 4] {
        let rotated = rotate(
            sharded(x.clone(), &sharded_compute(num_members), Split { dim: 2 }),
            angles.clone(),
        );

        assert_eq!(
            rotated.sharding(),
            Some(Split { dim: 1 }),
            "{num_members} members"
        );
        rotated
            .into_data()
            .assert_approx_eq::<f32>(&expected, tolerance());
    }
}

/// Rotary position embedding as a tensor-op formulation writes it: heads moved outward, then each
/// head's halves rotated into a zero buffer.
fn rotate(x: Tensor<4>, angles: TensorData) -> Tensor<4> {
    let heads = x.swap_dims(1, 2);
    let half = heads.dims()[3] / 2;
    let (lower, upper) = (
        heads.clone().narrow(3, 0, half),
        heads.clone().narrow(3, half, half),
    );
    let rotated = heads
        .zeros_like()
        .slice_assign(s![.., .., .., 0..half], upper.neg())
        .slice_assign(s![.., .., .., half..2 * half], lower);
    let angles = Tensor::<2>::from_data(angles, &heads.device()).unsqueeze::<4>();
    heads * angles.clone().cos() + rotated * angles.sin()
}

#[test]
fn attention_split_on_the_heads_stays_split() {
    let (query, key, value) = (
        values([2, 4, 3, 4], 1),
        values([2, 2, 5, 4], 2),
        values([2, 2, 5, 4], 3),
    );
    let flex = Device::flex();
    let expected = module::attention(
        Tensor::from_data(query.clone(), &flex),
        Tensor::from_data(key.clone(), &flex),
        Tensor::from_data(value.clone(), &flex),
        None,
        None,
        None,
        Default::default(),
    )
    .into_data();

    let device = sharded_compute(2);
    let heads = Split { dim: 1 };
    let context = module::attention(
        sharded::<4>(query, &device, heads),
        sharded(key, &device, heads),
        sharded(value, &device, heads),
        None,
        None,
        None,
        Default::default(),
    );

    assert_eq!(context.sharding(), Some(heads));
    context
        .into_data()
        .assert_approx_eq::<f32>(&expected, tolerance());
}

#[test]
fn attention_over_a_cache_split_on_the_heads_stays_split() {
    // Each sequence attends a different prefix of the cache, which is filled past the shortest.
    let lengths = TensorData::new(vec![5i64, 3], [2]);
    check_cached_attention_on_the_heads(
        values([2, 2, 6, 4], 2),
        values([2, 2, 6, 4], 3),
        lengths,
        None,
    );
}

#[test]
fn attention_over_a_paged_cache_split_on_the_heads_stays_split() {
    // Interleaved pages, so a member that reads another sequence's page is caught.
    let lengths = TensorData::new(vec![5i64, 3], [2]);
    let pages = TensorData::new(vec![0i64, 2, 4, 1, 3, 5], [2, 3]);
    check_cached_attention_on_the_heads(
        values([6, 2, 2, 4], 2),
        values([6, 2, 2, 4], 3),
        lengths,
        Some(pages),
    );
}

/// A decode step of four query heads over two key/value heads, on one device and with the query,
/// key and value split on the heads over two members: the context stays split and matches.
fn check_cached_attention_on_the_heads(
    key: TensorData,
    value: TensorData,
    lengths: TensorData,
    pages: Option<TensorData>,
) {
    let query = values([2, 4, 1, 4], 1);
    let cache = |device: &Device| {
        let lengths = Tensor::<1, Int>::from_data(lengths.clone(), device);
        match &pages {
            Some(pages) => module::AttentionCache::paged(
                lengths,
                Tensor::<2, Int>::from_data(pages.clone(), device),
            ),
            None => module::AttentionCache::new(lengths),
        }
    };
    let flex = Device::flex();
    let expected = module::attention(
        Tensor::from_data(query.clone(), &flex),
        Tensor::from_data(key.clone(), &flex),
        Tensor::from_data(value.clone(), &flex),
        None,
        None,
        Some(cache(&flex)),
        Default::default(),
    )
    .into_data();

    let device = sharded_compute(2);
    let heads = Split { dim: 1 };
    let context = module::attention(
        sharded::<4>(query, &device, heads),
        sharded(key, &device, heads),
        sharded(value, &device, heads),
        None,
        None,
        Some(cache(&device)),
        Default::default(),
    );

    assert_eq!(context.sharding(), Some(heads));
    context
        .into_data()
        .assert_approx_eq::<f32>(&expected, tolerance());
}

#[test]
fn a_split_reshaped_onto_an_empty_dim_adds_a_replicated_tensor() {
    let device = sharded_compute(2);
    let split = sharded::<2>(
        TensorData::new(Vec::<f32>::new(), [4, 0]),
        &device,
        Split { dim: 0 },
    );

    let sum = split.reshape([-1, 4]) + Tensor::zeros([0, 4], &device);

    sum.into_data()
        .assert_approx_eq::<f32>(&TensorData::new(Vec::<f32>::new(), [0, 4]), tolerance());
}

#[test]
fn int_and_bool_tensors_split_and_move_back_whole() {
    let ints = TensorData::from([[1i64, 2, 3, 4], [5, 6, 7, 8]]);
    let bools = TensorData::from([[true, false, true, true], [false, false, true, false]]);

    let split_ints = on_members::<2, Int>(ints.clone(), &sharded_compute(2), Split { dim: 1 });
    let split_bools = on_members::<2, Bool>(bools.clone(), &sharded_compute(2), Split { dim: 0 });

    assert_eq!(split_ints.sharding(), Some(Split { dim: 1 }));
    assert_eq!(split_bools.sharding(), Some(Split { dim: 0 }));
    split_ints.into_data().assert_eq(&ints, false);
    split_bools.into_data().assert_eq(&bools, false);
}

#[test]
fn a_quantized_weight_split_by_columns_keeps_every_value() {
    let device = sharded_compute(2);
    let weight = block_quantized(values([4, 8], 2));
    let x = values([3, 4], 1);
    let expected = Tensor::<2>::from_data(x.clone(), &Device::flex())
        .matmul(Tensor::from_data(weight.clone(), &Device::flex()))
        .into_data();

    let split = sharded::<2>(weight.clone(), &device, Split { dim: 1 });
    let product = Tensor::<2>::from_data(x, &device).matmul(split.clone());

    assert_eq!(split.into_data(), weight);
    assert_eq!(product.sharding(), Some(Split { dim: 1 }));
    product
        .into_data()
        .assert_approx_eq::<f32>(&expected, tolerance());
}

#[test]
fn a_quantized_weight_split_by_rows_gives_a_partial_sum() {
    let device = sharded_compute(2);
    let weight = block_quantized(values([8, 8], 2));
    let x = values([3, 8], 1);
    let expected = Tensor::<2>::from_data(x.clone(), &Device::flex())
        .matmul(Tensor::from_data(weight.clone(), &Device::flex()))
        .into_data();

    let split = sharded::<2>(weight, &device, Split { dim: 0 });
    let product = sharded::<2>(x, &device, Split { dim: 1 }).matmul(split);

    assert_eq!(product.sharding(), Some(PartialSum));
    product
        .into_data()
        .assert_approx_eq::<f32>(&expected, tolerance());
}

#[test]
fn a_split_quantized_weight_dequantizes_on_each_member() {
    let weight = block_quantized(values([4, 8], 2));
    let expected = Tensor::<2>::from_data(weight.clone(), &Device::flex())
        .dequantize()
        .into_data();

    let dequantized = sharded::<2>(weight, &sharded_compute(2), Split { dim: 1 }).dequantize();

    assert_eq!(dequantized.sharding(), Some(Split { dim: 1 }));
    dequantized
        .into_data()
        .assert_approx_eq::<f32>(&expected, tolerance());
}

#[test]
fn shards_join_into_one_tensor_and_come_back_as_they_were() {
    let device = sharded_compute(2);
    let members = device
        .shard_members()
        .expect("A sharded compute device has members");
    // Five rows over two members: the first holds three.
    let shards: Vec<Tensor<2>> = [values([3, 2], 1), values([2, 2], 2)]
        .into_iter()
        .zip(&members)
        .map(|(data, member)| Tensor::from_data(data, member))
        .collect();
    let expected: Vec<TensorData> = shards.iter().map(|shard| shard.to_data()).collect();

    let sharding = Split { dim: 0 };
    let joined = Tensor::from_shards(Shards { shards, sharding }, &device).unwrap();
    assert_eq!(joined.dims(), [5, 2]);
    let Shards { shards, sharding } = joined.into_shards();

    assert_eq!(sharding, Split { dim: 0 });
    for ((shard, expected), member) in shards.into_iter().zip(expected).zip(&members) {
        assert_eq!(shard.device(), *member);
        shard.into_data().assert_eq(&expected, true);
    }
}

#[test]
fn shards_that_are_not_the_balanced_split_are_refused() {
    let device = sharded_compute(2);
    let members = device.shard_members().unwrap();
    let shards = [values([2, 2], 1), values([3, 2], 2)]
        .into_iter()
        .zip(&members)
        .map(|(data, member)| Tensor::<2>::from_data(data, member))
        .collect();

    assert_eq!(
        Tensor::from_shards(
            Shards {
                shards,
                sharding: Split { dim: 0 },
            },
            &device
        )
        .err(),
        Some(ShardsError::Shape {
            shard: 0,
            expected: [3, 2].into(),
            found: [2, 2].into(),
        })
    );
}

#[test]
fn a_quantized_tensor_split_either_way_reads_back_unchanged() {
    let scheme = QuantScheme::default().per_block([4], ScaleDtype::F32);
    let whole = Tensor::<2>::from_data(values([4, 8], 1), &Device::flex())
        .quantize_dynamic(&scheme)
        .into_data();

    for sharding in [Split { dim: 0 }, Split { dim: 1 }] {
        let tensor = sharded::<2>(whole.clone(), &sharded_compute(2), sharding);

        assert_eq!(tensor.sharding(), Some(sharding));
        assert_eq!(tensor.into_data(), whole, "{sharding:?}");
    }
}

#[test]
fn a_quantized_tensor_on_a_sharded_compute_device_reports_its_sharding() {
    let quantized = Tensor::<2>::from_data(values([4, 8], 1), &sharded_compute(2))
        .quantize_dynamic(&QuantScheme::default());

    assert_eq!(quantized.sharding(), Some(Replicated));
}

#[test]
fn a_quantized_replicated_rhs_matmuls_with_a_split_lhs() {
    let device = sharded_compute(2);
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

    // Splitting the weight here would cut it on the host on every call, so the lhs gathers.
    assert_eq!(product.sharding(), Some(Replicated));
    product
        .into_data()
        .assert_approx_eq::<f32>(&expected, tolerance());
}

#[test]
fn a_tensor_moves_onto_a_sharded_compute_device_replicated_and_back_whole() {
    let data = values([4, 6], 1);
    let tensor =
        Tensor::<2>::from_data(data.clone(), &Device::flex()).to_device(&sharded_compute(2));
    assert_eq!(tensor.sharding(), Some(Replicated));

    let back = tensor.to_device(&Device::flex());
    assert_eq!(back.sharding(), None);
    back.into_data().assert_approx_eq::<f32>(&data, tolerance());
}

#[test]
fn a_split_tensor_moves_back_whole() {
    let data = values([4, 6], 1);

    let back = sharded::<2>(data.clone(), &sharded_compute(2), Split { dim: 1 })
        .to_device(&Device::flex());

    assert_eq!(back.sharding(), None);
    back.into_data().assert_approx_eq::<f32>(&data, tolerance());
}

#[test]
fn shards_taken_from_a_clone_leave_the_tensor_whole() {
    let data = values([4, 6], 1);
    let tensor = sharded::<2>(data.clone(), &sharded_compute(2), Split { dim: 1 });

    let Shards { shards, sharding } = tensor.clone().into_shards();

    assert_eq!(sharding, Split { dim: 1 });
    assert_eq!(shards.len(), 2);
    assert_eq!(tensor.sharding(), Some(Split { dim: 1 }));
    tensor
        .into_data()
        .assert_approx_eq::<f32>(&data, tolerance());
}

#[test]
fn a_split_leaf_made_of_shards_takes_its_gradient() {
    let device = sharded_compute(2).autodiff();
    let x = sharded::<1>(values([4], 1), &device, Split { dim: 0 }).require_grad();
    let y = x.clone().mul_scalar(2.0) + x.clone();
    let grads = y.sum().backward();

    let grad = x.grad(&grads).expect("x has a gradient");
    grad.into_data()
        .assert_approx_eq::<f32>(&TensorData::from([3.0f32; 4]), tolerance());
}

#[test]
fn a_random_tensor_is_one_draw_replicated_on_every_member() {
    let _rng = RNG.lock().unwrap_or_else(PoisonError::into_inner);
    let random = Tensor::<2>::random([4, 6], Distribution::Default, &sharded_compute(2));

    let Shards { shards, sharding } = random.into_shards();

    assert_eq!(sharding, Replicated);
    let first = shards[0].to_data();
    for shard in shards {
        shard.into_data().assert_eq(&first, true);
    }
}

#[test]
fn an_empty_sharded_compute_device_is_refused() {
    let refused = Device::shard_options(&[]).init();

    assert!(matches!(refused, Err(ShardError::NoDevices)));
}

#[test]
fn an_autodiff_member_after_the_first_is_refused_as_one() {
    let refused = Device::shard_options(&[Device::flex(), Device::flex().autodiff()]).init();

    assert!(matches!(refused, Err(ShardError::AutodiffMember)));
}

#[test]
fn ops_that_do_not_distribute_over_a_sum_reduce_a_partial_sum_first() {
    let (lhs, rhs) = (values([4, 6], 1), values([6, 3], 2));
    let flex = Device::flex();
    let whole =
        Tensor::<2>::from_data(lhs.clone(), &flex).matmul(Tensor::from_data(rhs.clone(), &flex));
    let expected_exp = whole.clone().exp().into_data();
    let expected_shifted = whole.add_scalar(1.0).into_data();

    let device = sharded_compute(2);
    let partial = sharded::<2>(lhs, &device, Split { dim: 1 }).matmul(sharded(
        rhs,
        &device,
        Split { dim: 0 },
    ));
    assert_eq!(partial.sharding(), Some(PartialSum));

    partial
        .clone()
        .exp()
        .into_data()
        .assert_approx_eq::<f32>(&expected_exp, tolerance());
    partial
        .add_scalar(1.0)
        .into_data()
        .assert_approx_eq::<f32>(&expected_shifted, tolerance());
}

#[test]
fn sharded_compute_devices_of_the_same_devices_are_apart() {
    let (first, second) = (sharded_compute(2), sharded_compute(2));
    assert_ne!(first, second);

    let data = values([4], 1);
    let tensor = sharded::<1>(data.clone(), &first, Split { dim: 0 });
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
            &sharded_compute(num_members),
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

fn sharded_compute(num_members: usize) -> Device {
    let members: Vec<Device> = member_devices()
        .into_iter()
        .cycle()
        .take(num_members)
        .collect();
    Device::shard_options(&members)
        .init()
        .expect("The test's devices form a sharded compute device")
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
    // Without both, the sharded compute device spans one runtime and the suite proves nothing it was run for.
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
    on_members(data, device, sharding)
}

/// `data` as `sharding` lays it over `device`, each member's shard made on it, or whole on a
/// device that is not a sharded compute device.
fn on_members<const D: usize, K: Autodiff>(
    data: TensorData,
    device: &Device,
    sharding: Sharding,
) -> Tensor<D, K> {
    let Some(members) = device.shard_members() else {
        return Tensor::from_data(data, device);
    };
    let chunks: Vec<TensorData> = match sharding {
        Split { dim } => DimSplit {
            dim,
            len: data.shape()[dim],
            members: members.len(),
        }
        .data_chunks(&data)
        .collect(),
        Replicated => vec![data; members.len()],
        PartialSum => unreachable!("The tests make a partial sum with an op"),
    };
    let shards = chunks
        .into_iter()
        .zip(&members)
        .map(|(chunk, member)| Tensor::from_data(chunk, member))
        .collect();
    Tensor::from_shards(Shards { shards, sharding }, device)
        .expect("The test's shards make up one tensor")
}

/// `data` quantized on the host with a scale per block of four along its last dim.
fn block_quantized(data: TensorData) -> TensorData {
    let scheme = QuantScheme::default().per_block([4], ScaleDtype::F32);
    Tensor::<2>::from_data(data, &Device::flex())
        .quantize_dynamic(&scheme)
        .into_data()
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
