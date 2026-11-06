use burn_backend::{
    Backend, BoolDType, Scalar, Shape, Slice, TensorData,
    ops::{ActivationOps, AttentionModuleOptions, DimSplit, FloatTensorOps, IntTensorOps},
    tensor::{FloatTensor, IntTensor},
};
use burn_flex::{Flex, FlexDevice};
use burn_ir::HandleKind;
use burn_shard::{
    ShardBackend, ShardedComputeDevice,
    Sharding::{self, PartialSum, Replicated, Split},
    Shards,
};
use burn_std::future::block_on;

type Shard = ShardBackend;

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
            |a| op::<Shard>(a, $($env.clone()),*),
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
            |a, b| op::<Shard>(a, b, $($env.clone()),*),
            |a, b| op::<Flex>(a, b, $($env.clone()),*),
        )
    }};
}

#[test]
fn a_sharded_compute_device_lists_its_members_in_order() {
    let device = sharded_compute(3);

    assert_eq!(device.members::<Flex>(), Some(vec![FlexDevice; 3]));
}

#[test]
fn elementwise_ops_match_one_device_for_every_sharding() {
    for num_members in 1..=4 {
        for rhs in [[4, 6], [1, 6], [4, 1]] {
            let (lhs, rhs) = (data([4, 6], 1), data(rhs, 2));
            check_binary!(num_members, &lhs, &rhs, |B, a, b| B::float_add(a, b));
            check_binary!(num_members, &lhs, &rhs, |B, a, b| B::float_sub(a, b));
            check_binary!(num_members, &lhs, &rhs, |B, a, b| B::float_mul(a, b));
            check_binary!(num_members, &lhs, &rhs, |B, a, b| B::relu_backward(a, b));
            check_binary!(num_members, &lhs, &positive(rhs), |B, a, b| B::float_div(
                a, b
            ));
        }
        let (lhs, rhs) = (data([4, 6], 1), data([4, 6], 2));
        check_binary!(num_members, &lhs, &rhs, |B, a, b| {
            let mask = B::float_greater_elem(b.clone(), Scalar::Float(0.0), BoolDType::Native);
            B::float_mask_where(a, mask, b)
        });
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
        check_unary!(num_members, &input, |B, a| B::float_add_scalar(
            a,
            Scalar::Float(3.0)
        ));
        check_unary!(num_members, &input, |B, a| B::float_clamp(
            a,
            Scalar::Float(-0.5),
            Scalar::Float(0.5)
        ));
        check_unary!(num_members, &input, |B, a| B::float_exp(a));
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
fn reductions_match_one_device_for_every_sharding() {
    for num_members in 1..=4 {
        let input = data([4, 3, 5], 1);
        for dim in 0..3 {
            check_unary!(num_members, &input, [dim: usize] |B, a| B::float_sum_dim(a, dim));
            check_unary!(num_members, &input, [dim: usize] |B, a| B::float_mean_dim(a, dim));
            check_unary!(num_members, &input, [dim: usize] |B, a| B::float_max_dim(a, dim));
        }
        check_unary!(num_members, &input, |B, a| B::float_sum(a));
        check_unary!(num_members, &input, |B, a| B::float_mean(a));
        check_unary!(num_members, &input, |B, a| B::float_max(a));
    }
}

/// Each member's summand is `[[1, 1], [1, 1], [0, 0]]` and the whole `[[2, 2], [2, 2], [0, 0]]`,
/// so dividing or averaging each summand truncates to zero where the whole does not.
#[test]
fn integer_division_and_mean_sum_a_partial_sum_first() {
    let device = sharded_compute(2);
    let int = |values: Vec<i64>, shape: [usize; 2], sharding| {
        made_on_members(&TensorData::new(values, shape), &device, sharding, |data| {
            HandleKind::Int(Flex::int_from_data(data, &FlexDevice))
        })
    };
    let partial = || {
        Shard::int_matmul(
            int(vec![1, 1, 1, 1, 0, 0], [3, 2], Split { dim: 1 }),
            int(vec![1; 4], [2, 2], Split { dim: 0 }),
        )
    };
    assert_eq!(sharding_of(&partial()), PartialSum);

    let divided = Shard::int_div(partial(), int(vec![2; 6], [3, 2], Replicated));
    let divided_by_scalar = Shard::int_div_scalar(partial(), Scalar::Int(2));
    let mean = Shard::int_mean_dim(partial(), 0);

    assert_eq!(read_ints(divided), [1, 1, 1, 1, 0, 0]);
    assert_eq!(read_ints(divided_by_scalar), [1, 1, 1, 1, 0, 0]);
    assert_eq!(read_ints(mean), [1, 1]);
}

#[test]
fn ops_along_a_dim_match_one_device_for_every_sharding() {
    for num_members in 1..=4 {
        let input = data([4, 3, 5], 1);
        for dim in 0..3 {
            check_unary!(num_members, &input, [dim: usize] |B, a| B::softmax(a, dim));
            check_unary!(num_members, &input, [dim: usize] |B, a| B::float_cumsum(a, dim));
            check_unary!(num_members, &input, [dim: usize] |B, a| B::float_topk(a, dim, 2));
        }
    }
}

/// Each slice takes dim 0 whole and cuts dim 1, so a split on dim 0 stays and one on dim 1
/// gathers.
#[test]
fn layout_ops_match_one_device_for_every_sharding() {
    let ranges = [0..4, 1..3, 0..5].map(Slice::from);
    for num_members in 1..=4 {
        let input = data([4, 3, 5], 1);
        check_unary!(num_members, &input, |B, a| B::float_swap_dims(a, 0, 2));
        check_unary!(num_members, &input, |B, a| B::float_permute(a, &[2, 0, 1]));
        check_unary!(num_members, &data([4, 1, 5], 2), |B, a| B::float_expand(
            a,
            Shape::new([2, 4, 3, 5])
        ));
        check_unary!(num_members, &input, [ranges: [Slice; 3]] |B, a| B::float_slice(a, &ranges));
        check_binary!(num_members, &input, &data([4, 2, 5], 3), [ranges: [Slice; 3]] |B, a, b| {
            B::float_slice_assign(a, &ranges, b)
        });
    }
}

/// Heads that split unevenly, grouped query heads, and one key and value head shared by all.
#[test]
fn attention_matches_one_device_for_every_sharding() {
    for num_members in 1..=4 {
        for (q_heads, kv_heads) in [(3, 3), (4, 2), (4, 1)] {
            let inputs = AttentionInputs {
                query: data([2, q_heads, 3, 4], 1),
                key: data([2, kv_heads, 5, 4], 2),
                value: data([2, kv_heads, 5, 4], 3),
                mask_and_bias: None,
            };
            inputs.assert_matches(num_members, AttentionModuleOptions::default());
        }
    }
}

/// The mask is split along the heads with the query, and the bias, broadcast over them, is not.
#[test]
fn masked_causal_attention_matches_one_device_for_every_sharding() {
    let mask = (0..2 * 4 * 3 * 5).map(|i| i % 7 == 3).collect::<Vec<_>>();
    let inputs = AttentionInputs {
        query: data([2, 4, 3, 4], 1),
        key: data([2, 2, 5, 4], 2),
        value: data([2, 2, 5, 4], 3),
        mask_and_bias: Some((TensorData::new(mask, [2, 4, 3, 5]), data([2, 1, 3, 5], 4))),
    };
    let options = AttentionModuleOptions {
        is_causal: true,
        ..Default::default()
    };
    for num_members in 1..=4 {
        inputs.assert_matches(num_members, options);
    }
}

#[test]
fn split_tensors_stay_split_where_megatron_keeps_them() {
    let device = sharded_compute(2);
    let x = float_on_members(&data([4, 6], 1), &device, Split { dim: 1 });
    let w = float_on_members(&data([6, 6], 2), &device, Split { dim: 0 });
    let summand = Shard::float_matmul(x.clone(), w);

    assert_eq!(sharding_of(&summand), PartialSum);
    assert_eq!(
        sharding_of(&Shard::float_add(summand.clone(), summand)),
        PartialSum
    );
    assert_eq!(sharding_of(&Shard::float_sum_dim(x.clone(), 1)), PartialSum);
    assert_eq!(sharding_of(&Shard::relu(x.clone())), Split { dim: 1 });
    assert_eq!(sharding_of(&Shard::float_exp(x.clone())), Split { dim: 1 });
    assert_eq!(sharding_of(&Shard::softmax(x, 0)), Split { dim: 1 });
}

fn assert_unary_matches(
    num_members: usize,
    input: &TensorData,
    on_shards: impl Fn(FloatTensor<Shard>) -> FloatTensor<Shard>,
    on_flex: impl Fn(FloatTensor<Flex>) -> FloatTensor<Flex>,
) {
    let device = sharded_compute(num_members);
    for sharding in shardings(input.shape(), num_members) {
        let (tensor, value) = sharded(input, &device, sharding);
        let expected = read::<Flex>(on_flex(Flex::float_from_data(value, &FlexDevice)));
        let actual = read::<Shard>(on_shards(tensor));
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
    on_shards: impl Fn(FloatTensor<Shard>, FloatTensor<Shard>) -> FloatTensor<Shard>,
    on_flex: impl Fn(FloatTensor<Flex>, FloatTensor<Flex>) -> FloatTensor<Flex>,
) {
    let device = sharded_compute(num_members);
    for lhs_sharding in shardings(lhs.shape(), num_members) {
        for rhs_sharding in shardings(rhs.shape(), num_members) {
            let (lhs_tensor, lhs_value) = sharded(lhs, &device, lhs_sharding);
            let (rhs_tensor, rhs_value) = sharded(rhs, &device, rhs_sharding);
            let expected = read::<Flex>(on_flex(
                Flex::float_from_data(lhs_value, &FlexDevice),
                Flex::float_from_data(rhs_value, &FlexDevice),
            ));
            let actual = read::<Shard>(on_shards(lhs_tensor, rhs_tensor));
            let case = format!("{num_members} members, {lhs_sharding:?} with {rhs_sharding:?}");
            assert_close(actual, expected, &case);
        }
    }
}

struct AttentionInputs {
    query: TensorData,
    key: TensorData,
    value: TensorData,
    mask_and_bias: Option<(TensorData, TensorData)>,
}

impl AttentionInputs {
    /// For every sharding of the query, and of the key and value together, the mask and bias
    /// replicated.
    fn assert_matches(&self, num_members: usize, options: AttentionModuleOptions) {
        let device = sharded_compute(num_members);
        let expected = read::<Flex>(
            self.attention::<Flex>(
                [&self.query, &self.key, &self.value]
                    .map(|data| Flex::float_from_data(data.clone(), &FlexDevice)),
                &FlexDevice,
                options,
            ),
        );
        for query_sharding in shardings(self.query.shape(), num_members) {
            for kv_sharding in shardings(self.key.shape(), num_members) {
                let (query, query_value) = sharded(&self.query, &device, query_sharding);
                let (key, key_value) = sharded(&self.key, &device, kv_sharding);
                let (value, value_value) = sharded(&self.value, &device, kv_sharding);
                let case = format!(
                    "{num_members} members, query {query_sharding:?}, key and value {kv_sharding:?}"
                );
                // A partial sum stands for a contraction's product, which changes the expected
                // value: those cases are checked against one device run on that product.
                let expected = match (query_sharding, kv_sharding) {
                    (PartialSum, _) | (_, PartialSum) => read::<Flex>(
                        self.attention::<Flex>(
                            [query_value, key_value, value_value]
                                .map(|data| Flex::float_from_data(data, &FlexDevice)),
                            &FlexDevice,
                            options,
                        ),
                    ),
                    _ => expected.clone(),
                };
                let actual =
                    read::<Shard>(self.attention::<Shard>([query, key, value], &device, options));
                assert_close(actual, expected, &case);
            }
        }
    }

    fn attention<B: Backend>(
        &self,
        [query, key, value]: [FloatTensor<B>; 3],
        device: &B::Device,
        options: AttentionModuleOptions,
    ) -> FloatTensor<B> {
        let (mask, bias) = match &self.mask_and_bias {
            Some((mask, bias)) => (
                Some(B::bool_from_data(mask.clone(), device)),
                Some(B::float_from_data(bias.clone(), device)),
            ),
            None => (None, None),
        };
        B::attention(query, key, value, mask, bias, None, options)
    }
}

/// `data` with the sharding, and the value one device gives it: a partial sum is built as the
/// contraction `data @ weights` with both split, so it holds the product instead.
fn sharded(
    data: &TensorData,
    device: &ShardedComputeDevice,
    sharding: Sharding,
) -> (FloatTensor<Shard>, TensorData) {
    if sharding != PartialSum {
        return (float_on_members(data, device, sharding), data.clone());
    }
    let num_dims = data.shape().num_dims();
    let weights = contraction_weights(data.shape());
    let partial = Shard::float_matmul(
        float_on_members(data, device, Split { dim: num_dims - 1 }),
        float_on_members(&weights, device, Split { dim: num_dims - 2 }),
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

/// A partial sum needs a last dim to contract, at least as long as the sharded compute device.
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

fn sharded_compute(num_members: usize) -> ShardedComputeDevice {
    ShardedComputeDevice::new::<Flex>(&vec![FlexDevice; num_members])
        .expect("Flex devices form a sharded compute device")
}

/// `data` replicated or split over `device`, each member's shard made on it.
fn float_on_members(
    data: &TensorData,
    device: &ShardedComputeDevice,
    sharding: Sharding,
) -> FloatTensor<Shard> {
    made_on_members(data, device, sharding, |data| {
        HandleKind::Float(Flex::float_from_data(data, &FlexDevice))
    })
}

/// `data` replicated or split over `device`, each member's shard uploaded by `upload`. A float
/// and an int tensor of the sharded compute device are one type.
fn made_on_members(
    data: &TensorData,
    device: &ShardedComputeDevice,
    sharding: Sharding,
    upload: impl Fn(TensorData) -> HandleKind<Flex>,
) -> FloatTensor<Shard> {
    let num_members = device.num_members();
    let chunks: Vec<TensorData> = match sharding {
        Split { dim } => DimSplit {
            dim,
            len: data.shape()[dim],
            members: num_members,
        }
        .data_chunks(data)
        .collect(),
        Replicated => vec![data.clone(); num_members],
        PartialSum => unreachable!("The tests make a partial sum with a contraction"),
    };
    device
        .join_shards::<Flex>(Shards {
            shards: chunks.into_iter().map(upload).collect(),
            sharding,
        })
        .expect("The test's shards make up one tensor")
}

fn sharding_of(tensor: &FloatTensor<Shard>) -> Sharding {
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

fn read_ints(tensor: IntTensor<Shard>) -> Vec<i64> {
    block_on(Shard::int_into_data(tensor))
        .expect("The tensor can be read")
        .iter::<i64>()
        .collect()
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
