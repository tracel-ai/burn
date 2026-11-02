use burn_backend::{
    Backend, Distribution, FloatDType, Scalar, Shape, TensorData,
    ops::{ActivationOps, FloatTensorOps, QTensorOps},
    quantization::QuantScheme,
    tensor::FloatTensor,
};
use burn_flex::{Flex, FlexDevice};
use burn_group::{
    GroupBackend, GroupDevice,
    Sharding::{self, PartialSum, Replicated, Split},
    ShardingError,
};
use burn_std::future::block_on;
use std::sync::{Mutex, PoisonError};

type Group = GroupBackend;

/// Flex keeps one RNG for the whole process, so tests that seed it take turns.
static RNG: Mutex<()> = Mutex::new(());

macro_rules! check_unary {
    ($num_members:expr, $input:expr, |$backend:ident, $a:ident| $body:expr) => {
        check_unary!($num_members, $input, [] |$backend, $a| $body)
    };
    ($num_members:expr, $input:expr, [$($env:ident: $ty:ty),*] |$backend:ident, $a:ident| $body:expr) => {{
        fn op<$backend: Backend>($a: FloatTensor<$backend>, $($env: $ty),*) -> FloatTensor<$backend> {
            $body
        }
        assert_unary_matches(
            $num_members,
            $input,
            |a| op::<Group>(a, $($env.clone()),*),
            |a| op::<Flex>(a, $($env.clone()),*),
        )
    }};
}

macro_rules! check_binary {
    ($num_members:expr, $lhs:expr, $rhs:expr, |$backend:ident, $a:ident, $b:ident| $body:expr) => {
        check_binary!($num_members, $lhs, $rhs, [] |$backend, $a, $b| $body)
    };
    ($num_members:expr, $lhs:expr, $rhs:expr, [$($env:ident: $ty:ty),*] |$backend:ident, $a:ident, $b:ident| $body:expr) => {{
        fn op<$backend: Backend>(
            $a: FloatTensor<$backend>,
            $b: FloatTensor<$backend>,
            $($env: $ty),*
        ) -> FloatTensor<$backend> {
            $body
        }
        assert_binary_matches(
            $num_members,
            $lhs,
            $rhs,
            |a, b| op::<Group>(a, b, $($env.clone()),*),
            |a, b| op::<Flex>(a, b, $($env.clone()),*),
        )
    }};
}

#[test]
fn elementwise_ops_match_one_device_for_every_sharding() {
    for num_members in 1..=4 {
        for rhs in [[4, 6], [1, 6], [4, 1]] {
            let (lhs, rhs) = (data([4, 6], 1), data(rhs, 2));
            check_binary!(num_members, &lhs, &rhs, |B, a, b| B::float_add(a, b));
            check_binary!(num_members, &lhs, &rhs, |B, a, b| B::float_sub(a, b));
            check_binary!(num_members, &lhs, &rhs, |B, a, b| B::float_mul(a, b));
            check_binary!(num_members, &lhs, &positive(rhs), |B, a, b| B::float_div(
                a, b
            ));
        }
        let input = data([4, 6], 3);
        check_unary!(num_members, &input, |B, a| B::float_neg(a));
        check_unary!(num_members, &input, |B, a| B::float_mul_scalar(
            a,
            Scalar::Float(3.0)
        ));
        check_unary!(num_members, &input, |B, a| B::float_div_scalar(
            a,
            Scalar::Float(3.0)
        ));
        check_unary!(num_members, &input, |B, a| B::relu(a));
        check_unary!(num_members, &input, |B, a| B::gelu(a));
        check_unary!(num_members, &input, |B, a| B::sigmoid(a));
    }
}

#[test]
fn matmuls_match_one_device_for_every_sharding() {
    for num_members in 1..=4 {
        for (lhs, rhs) in [
            (vec![4, 6], vec![6, 5]),
            (vec![3, 4, 6], vec![3, 6, 5]),
            (vec![3, 4, 6], vec![1, 6, 5]),
        ] {
            let (lhs, rhs) = (data(lhs, 1), data(rhs, 2));
            check_binary!(num_members, &lhs, &rhs, |B, a, b| B::float_matmul(a, b));
        }
    }
}

#[test]
fn sums_match_one_device_for_every_sharding() {
    for num_members in 1..=4 {
        let input = data([4, 3, 5], 1);
        for dim in 0..3 {
            check_unary!(num_members, &input, [dim: usize] |B, a| B::float_sum_dim(a, dim));
        }
        check_unary!(num_members, &input, |B, a| B::float_sum(a));
    }
}

#[test]
fn split_tensors_stay_split_where_megatron_keeps_them() {
    let device = group(2);
    let x = reshard(
        Group::float_from_data(data([4, 6], 1), &device),
        Split { dim: 1 },
    );
    let w = reshard(
        Group::float_from_data(data([6, 6], 2), &device),
        Split { dim: 0 },
    );
    let summand = Group::float_matmul(x.clone(), w);

    assert_eq!(sharding_of(&summand), PartialSum);
    assert_eq!(
        sharding_of(&Group::float_add(summand.clone(), summand)),
        PartialSum
    );
    assert_eq!(sharding_of(&Group::float_sum_dim(x.clone(), 1)), PartialSum);
    assert_eq!(sharding_of(&Group::relu(x)), Split { dim: 1 });
}

#[test]
fn a_quantized_tensor_is_not_split() {
    // CUDA packs four quantized values per u32, so the last dim is a multiple of four.
    let whole = Group::float_from_data(data([4, 8], 1), &group(2));
    let quantized = Group::quantize_dynamic(whole, &QuantScheme::default());
    let client = quantized.client.clone();

    let refused = client.reshard(quantized, Split { dim: 0 });

    assert!(matches!(refused, Err(ShardingError::Quantized)));
}

#[test]
fn each_member_of_a_split_random_tensor_draws_its_own_shard() {
    let _rng = RNG.lock().unwrap_or_else(PoisonError::into_inner);
    let device = group(2);
    Group::seed(&device, 7);
    let split = read::<Group>(reshard(random::<Group>([4, 6], &device), Split { dim: 1 }));

    Flex::seed(&FlexDevice, 7);
    let shards = vec![
        random::<Flex>([4, 3], &FlexDevice),
        random::<Flex>([4, 3], &FlexDevice),
    ];

    // Split on dim 0, a whole draw cut in two has the same values as two shard draws.
    assert_eq!(split, read::<Flex>(Flex::float_cat(shards, 1)));
}

#[test]
fn a_seed_runs_the_random_tensors_made_before_it() {
    let _rng = RNG.lock().unwrap_or_else(PoisonError::into_inner);
    let device = group(2);
    Group::seed(&device, 7);
    let made_before = [
        random::<Group>([4, 6], &device),
        random::<Group>([4, 6], &device),
    ];
    Group::seed(&device, 7);
    let made_after = [
        random::<Group>([4, 6], &device),
        random::<Group>([4, 6], &device),
    ];

    assert_eq!(
        made_before.map(read::<Group>),
        made_after.map(read::<Group>)
    );
}

#[test]
fn every_sharding_moves_to_every_other_and_keeps_its_value() {
    let input = data([4, 6], 1);
    for num_members in 1..=4 {
        let group = group(num_members);
        for from in shardings(input.shape(), num_members) {
            for to in shardings(input.shape(), num_members) {
                if to == PartialSum && from != PartialSum {
                    continue;
                }
                let (tensor, value) = sharded(&input, &group, from);
                let moved = reshard(tensor, to);
                assert_eq!(sharding_of(&moved), to);
                let case = format!("{num_members} members, {from:?} to {to:?}");
                assert_close(read::<Group>(moved), value, &case);
            }
        }
    }
}

fn assert_unary_matches(
    num_members: usize,
    input: &TensorData,
    on_group: impl Fn(FloatTensor<Group>) -> FloatTensor<Group>,
    on_flex: impl Fn(FloatTensor<Flex>) -> FloatTensor<Flex>,
) {
    let device = group(num_members);
    for sharding in shardings(input.shape(), num_members) {
        let (tensor, value) = sharded(input, &device, sharding);
        let expected = read::<Flex>(on_flex(Flex::float_from_data(value, &FlexDevice)));
        let actual = read::<Group>(on_group(tensor));
        assert_close(
            actual,
            expected,
            &format!("{num_members} members, {sharding:?}"),
        );
    }
}

fn assert_binary_matches(
    num_members: usize,
    lhs: &TensorData,
    rhs: &TensorData,
    on_group: impl Fn(FloatTensor<Group>, FloatTensor<Group>) -> FloatTensor<Group>,
    on_flex: impl Fn(FloatTensor<Flex>, FloatTensor<Flex>) -> FloatTensor<Flex>,
) {
    let device = group(num_members);
    for lhs_sharding in shardings(lhs.shape(), num_members) {
        for rhs_sharding in shardings(rhs.shape(), num_members) {
            let (lhs_tensor, lhs_value) = sharded(lhs, &device, lhs_sharding);
            let (rhs_tensor, rhs_value) = sharded(rhs, &device, rhs_sharding);
            let expected = read::<Flex>(on_flex(
                Flex::float_from_data(lhs_value, &FlexDevice),
                Flex::float_from_data(rhs_value, &FlexDevice),
            ));
            let actual = read::<Group>(on_group(lhs_tensor, rhs_tensor));
            let case = format!("{num_members} members, {lhs_sharding:?} with {rhs_sharding:?}");
            assert_close(actual, expected, &case);
        }
    }
}

/// `data` with the sharding, and the value one device gives it: a partial sum is built as the
/// contraction `data @ weights` with both split, so it holds the product instead.
fn sharded(
    data: &TensorData,
    device: &GroupDevice,
    sharding: Sharding,
) -> (FloatTensor<Group>, TensorData) {
    let tensor = Group::float_from_data(data.clone(), device);
    if sharding != PartialSum {
        return (reshard(tensor, sharding), data.clone());
    }
    let num_dims = data.shape().num_dims();
    let weights = contraction_weights(data.shape());
    let partial = Group::float_matmul(
        reshard(tensor, Split { dim: num_dims - 1 }),
        reshard(
            Group::float_from_data(weights.clone(), device),
            Split { dim: num_dims - 2 },
        ),
    );
    assert_eq!(sharding_of(&partial), PartialSum);
    (partial, product_on_one_device(data, &weights))
}

/// Dense, so every member holds a summand of every element and a rule that wrongly keeps a value
/// partial, or a slice standing in for a reduce-scatter, comes out wrong. Positive, so a partial
/// sum of positive values stays away from zero and a division by one is bounded.
fn contraction_weights(shape: &Shape) -> TensorData {
    let num_dims = shape.num_dims();
    let len = shape[num_dims - 1];
    let mut weights_shape = vec![1; num_dims - 2];
    weights_shape.extend([len, len]);
    positive(data(weights_shape, 4))
}

fn product_on_one_device(data: &TensorData, weights: &TensorData) -> TensorData {
    read::<Flex>(Flex::float_matmul(
        Flex::float_from_data(data.clone(), &FlexDevice),
        Flex::float_from_data(weights.clone(), &FlexDevice),
    ))
}

/// A partial sum needs a last dim to contract, at least as long as the group.
fn shardings(shape: &Shape, num_members: usize) -> Vec<Sharding> {
    let num_dims = shape.num_dims();
    let mut shardings = vec![Replicated];
    if num_dims >= 2 && shape[num_dims - 1] >= num_members {
        shardings.push(PartialSum);
    }
    shardings.extend(
        (0..num_dims)
            .filter(|dim| shape[*dim] >= num_members)
            .map(|dim| Split { dim }),
    );
    shardings
}

fn group(num_members: usize) -> GroupDevice {
    GroupDevice::new::<Flex>(&vec![FlexDevice; num_members]).expect("Flex devices form a group")
}

fn reshard(tensor: FloatTensor<Group>, sharding: Sharding) -> FloatTensor<Group> {
    tensor
        .client
        .clone()
        .reshard(tensor, sharding)
        .expect("The test asks for a sharding the tensor can take")
}

fn sharding_of(tensor: &FloatTensor<Group>) -> Sharding {
    tensor.client.sharding(tensor)
}

fn data<S: Into<Shape>>(shape: S, seed: usize) -> TensorData {
    let shape = shape.into();
    let values: Vec<f32> = (0..shape.num_elements())
        .map(|i| ((i * 7 + seed * 13) as f32 * 0.37).sin())
        .collect();
    TensorData::new(values, shape)
}

fn positive(data: TensorData) -> TensorData {
    let values: Vec<f32> = data.iter::<f32>().map(|value| value.abs() + 1.0).collect();
    TensorData::new(values, data.shape().clone())
}

fn random<B: Backend>(shape: [usize; 2], device: &B::Device) -> FloatTensor<B> {
    B::float_random(shape.into(), Distribution::Default, device, FloatDType::F32)
}

fn read<B: Backend>(tensor: FloatTensor<B>) -> TensorData {
    block_on(B::float_into_data(tensor)).expect("The tensor can be read")
}

fn assert_close(actual: TensorData, expected: TensorData, case: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{case}");
    for (actual, expected) in actual.iter::<f32>().zip(expected.iter::<f32>()) {
        let bound = 1e-5 + 1e-4 * expected.abs();
        assert!(
            (actual - expected).abs() <= bound,
            "{case}: {actual} against {expected}"
        );
    }
}
