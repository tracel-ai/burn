use burn_autodiff::Autodiff;
use burn_backend::{
    AutodiffBackend, Backend, DType, Scalar, Shape, Slice, TensorData, TensorMetadata,
    ops::{FloatTensorOps, IntTensorOps},
    tensor::{FloatTensor, IntTensor},
};
use burn_flex::{Flex, FlexDevice};
use burn_group::{
    GroupBackend, GroupDevice,
    GroupPlacement::{self, Partial, Replicated, Sharded},
};
use burn_ir::{
    BinaryOpIr, GraphBindings, GraphId, NumericOperationIr, OperationIr, TensorId, TensorIr,
    TensorStatus,
};
use burn_router::{RouterClient, RouterTensor};
use burn_std::future::block_on;

type Group = GroupBackend;

const RANKS: [usize; 4] = [1, 2, 3, 4];

macro_rules! check_unary {
    ($ranks:expr, $input:expr, |$backend:ident, $a:ident| $body:expr) => {
        check_unary!($ranks, $input, [] |$backend, $a| $body)
    };
    ($ranks:expr, $input:expr, [$($env:ident: $ty:ty),*] |$backend:ident, $a:ident| $body:expr) => {{
        fn op<$backend: Backend>($a: FloatTensor<$backend>, $($env: $ty),*) -> FloatTensor<$backend> {
            $body
        }
        assert_unary_matches(
            $ranks,
            $input,
            |a| op::<Group>(a, $($env.clone()),*),
            |a| op::<Flex>(a, $($env.clone()),*),
        )
    }};
}

macro_rules! check_binary {
    ($ranks:expr, $lhs:expr, $rhs:expr, |$backend:ident, $a:ident, $b:ident| $body:expr) => {
        check_binary!($ranks, $lhs, $rhs, [] |$backend, $a, $b| $body)
    };
    ($ranks:expr, $lhs:expr, $rhs:expr, [$($env:ident: $ty:ty),*] |$backend:ident, $a:ident, $b:ident| $body:expr) => {{
        fn op<$backend: Backend>(
            $a: FloatTensor<$backend>,
            $b: FloatTensor<$backend>,
            $($env: $ty),*
        ) -> FloatTensor<$backend> {
            $body
        }
        assert_binary_matches(
            $ranks,
            $lhs,
            $rhs,
            |a, b| op::<Group>(a, b, $($env.clone()),*),
            |a, b| op::<Flex>(a, b, $($env.clone()),*),
        )
    }};
}

#[test]
fn megatron_mlp_matches_one_device() {
    for (ranks, hidden) in [(1, 12), (2, 12), (3, 12), (4, 12), (4, 10)] {
        let device = group(ranks);
        let (x, target) = (data([4, 6], 1), data([4, 5], 2));
        let (w1, b1) = (data([6, hidden], 3), data([hidden], 4));
        let (w2, b2) = (data([hidden, 5], 5), data([5], 6));

        let expected = Mlp::run::<Autodiff<Flex>>(
            [&x, &w1, &b1, &w2, &b2].map(|data| leaf(data.clone())),
            Autodiff::<Flex>::float_from_data(target.clone(), &FlexDevice),
        );
        let actual = Mlp::run::<Autodiff<Group>>(
            [
                placed_leaf(&x, &device, Replicated),
                placed_leaf(&w1, &device, Sharded { dim: 1 }),
                placed_leaf(&b1, &device, Sharded { dim: 0 }),
                placed_leaf(&w2, &device, Sharded { dim: 0 }),
                placed_leaf(&b2, &device, Replicated),
            ],
            Autodiff::<Group>::float_from_data(target, &device),
        );

        let grad_placements: Vec<GroupPlacement> = actual.grads.iter().map(placement_of).collect();
        assert_eq!(
            grad_placements,
            [
                Partial,
                Sharded { dim: 1 },
                Sharded { dim: 0 },
                Sharded { dim: 0 },
                Replicated
            ],
            "{ranks} ranks"
        );
        actual.assert_matches(expected, &format!("{ranks} ranks"));
    }
}

#[test]
fn megatron_attention_matches_one_device() {
    for ranks in RANKS {
        let device = group(ranks);
        let x = data([2, 5, Attention::EMBED], 1);
        let weights = [2, 3, 4, 5].map(|seed| data([Attention::EMBED, Attention::EMBED], seed));

        let expected = Attention::run::<Autodiff<Flex>>(leaf(x.clone()), weights.clone().map(leaf));
        let [wq, wk, wv, wo] = &weights;
        let actual = Attention::run::<Autodiff<Group>>(
            placed_leaf(&x, &device, Replicated),
            [
                placed_leaf(wq, &device, Sharded { dim: 1 }),
                placed_leaf(wk, &device, Sharded { dim: 1 }),
                placed_leaf(wv, &device, Sharded { dim: 1 }),
                placed_leaf(wo, &device, Sharded { dim: 0 }),
            ],
        );

        if Attention::HEADS.is_multiple_of(ranks) {
            let weight_placements: Vec<GroupPlacement> =
                actual.grads[1..].iter().map(placement_of).collect();
            assert_eq!(
                weight_placements,
                [
                    Sharded { dim: 1 },
                    Sharded { dim: 1 },
                    Sharded { dim: 1 },
                    Sharded { dim: 0 }
                ],
                "{ranks} ranks"
            );
        }
        actual.assert_matches(expected, &format!("{ranks} ranks"));
    }
}

#[test]
fn embedding_matches_one_device_for_every_split() {
    let (weights, indices) = (data([7, 4], 1), TensorData::from([[0i64, 6, 3], [2, 2, 5]]));
    let scale = data([2, 3, 4], 2);
    for ranks in [1, 2, 3] {
        let device = group(ranks);
        for weights_placement in [Replicated, Sharded { dim: 0 }, Sharded { dim: 1 }] {
            for indices_placement in [Replicated, Sharded { dim: 1 }] {
                let expected = embedding::<Autodiff<Flex>>(
                    leaf(weights.clone()),
                    Autodiff::<Flex>::int_from_data(indices.clone(), &FlexDevice),
                    Autodiff::<Flex>::float_from_data(scale.clone(), &FlexDevice),
                );
                let placed_indices = place(
                    Group::int_from_data(indices.clone(), &device),
                    indices_placement,
                );
                let actual = embedding::<Autodiff<Group>>(
                    placed_leaf(&weights, &device, weights_placement),
                    Autodiff::<Group>::int_from_inner(placed_indices),
                    Autodiff::<Group>::float_from_data(scale.clone(), &device),
                );
                let case = format!(
                    "{ranks} ranks, {weights_placement:?} weights, {indices_placement:?} indices"
                );
                actual.assert_matches(expected, &case);
            }
        }
    }
}

#[test]
fn elementwise_ops_match_one_device_for_every_placement() {
    for ranks in RANKS {
        for rhs in [[4, 6], [1, 6], [4, 1]] {
            let (lhs, rhs) = (data([4, 6], 1), data(rhs, 2));
            check_binary!(ranks, &lhs, &rhs, |B, a, b| B::float_add(a, b));
            check_binary!(ranks, &lhs, &rhs, |B, a, b| B::float_sub(a, b));
            check_binary!(ranks, &lhs, &rhs, |B, a, b| B::float_mul(a, b));
            check_binary!(ranks, &lhs, &positive(rhs), |B, a, b| B::float_div(a, b));
        }
        let input = data([4, 6], 3);
        check_unary!(ranks, &input, |B, a| B::float_neg(a));
        check_unary!(ranks, &input, |B, a| B::float_exp(a));
        check_unary!(ranks, &input, |B, a| B::float_mul_scalar(
            a,
            Scalar::Float(3.0)
        ));
        check_unary!(ranks, &input, |B, a| B::float_div_scalar(
            a,
            Scalar::Float(3.0)
        ));
        check_unary!(ranks, &input, |B, a| B::float_add_scalar(
            a,
            Scalar::Float(3.0)
        ));
        check_unary!(ranks, &input, |B, a| B::relu(a));
    }
}

#[test]
fn matmuls_match_one_device_for_every_placement() {
    for ranks in RANKS {
        for (lhs, rhs) in [
            (vec![4, 6], vec![6, 5]),
            (vec![3, 4, 6], vec![3, 6, 5]),
            (vec![3, 4, 6], vec![1, 6, 5]),
        ] {
            let (lhs, rhs) = (data(lhs, 1), data(rhs, 2));
            check_binary!(ranks, &lhs, &rhs, |B, a, b| B::float_matmul(a, b));
        }
        let (x, weight, bias) = (data([2, 4, 6], 1), data([6, 5], 2), data([5], 3));
        check_binary!(ranks, &x, &weight, |B, x, weight| B::linear(
            x, weight, None
        ));
        check_binary!(ranks, &x, &weight, [bias: TensorData] |B, x, weight| {
            let device = x.device();
            B::linear(x, weight, Some(B::float_from_data(bias, &device)))
        });
    }
}

#[test]
fn reductions_and_softmax_match_one_device_for_every_placement() {
    for ranks in RANKS {
        let input = data([4, 3, 5], 1);
        for dim in 0..3 {
            check_unary!(ranks, &input, [dim: usize] |B, a| B::float_sum_dim(a, dim));
            check_unary!(ranks, &input, [dim: usize] |B, a| B::float_mean_dim(a, dim));
            check_unary!(ranks, &input, [dim: usize] |B, a| B::softmax(a, dim));
            check_unary!(ranks, &input, [dim: usize] |B, a| B::float_max_dim(a, dim));
        }
        check_unary!(ranks, &input, |B, a| B::float_sum(a));
        check_unary!(ranks, &input, |B, a| B::float_mean(a));
    }
}

#[test]
fn layout_ops_match_one_device_for_every_placement() {
    for ranks in RANKS {
        let input = data([4, 6], 1);
        for shape in [
            vec![24],
            vec![2, 2, 6],
            vec![4, 2, 3],
            vec![2, 12],
            vec![4, 1, 6],
        ] {
            let shape = Shape::from(shape);
            check_unary!(ranks, &input, [shape: Shape] |B, a| B::float_reshape(a, shape));
        }
        check_unary!(ranks, &input, |B, a| B::float_swap_dims(a, 0, 1));
        check_unary!(ranks, &input, |B, a| B::float_expand(
            a,
            Shape::new([3, 4, 6])
        ));
        check_unary!(ranks, &input, |B, a| B::float_slice(
            a,
            &[Slice::new(1, Some(3), 1)]
        ));
        check_unary!(ranks, &input, |B, a| {
            B::float_slice(a, &[Slice::new(0, None, 1), Slice::new(2, Some(5), 1)])
        });
        let other = data([4, 6], 2);
        check_binary!(ranks, &input, &other, |B, a, b| B::float_cat(vec![a, b], 0));
        check_binary!(ranks, &input, &other, |B, a, b| B::float_cat(vec![a, b], 1));
        let rank3 = data([2, 6, 4], 3);
        check_unary!(ranks, &rank3, |B, a| B::float_permute(a, &[2, 0, 1]));
        check_unary!(ranks, &rank3, |B, a| B::float_reshape(
            a,
            Shape::new([12, 4])
        ));
    }
}

#[test]
fn split_tensors_stay_split_where_megatron_keeps_them() {
    let device = group(2);
    let x = place(
        Group::float_from_data(data([4, 6], 1), &device),
        Sharded { dim: 1 },
    );
    let w = place(
        Group::float_from_data(data([6, 6], 2), &device),
        Sharded { dim: 0 },
    );
    let summand = Group::float_matmul(x.clone(), w);

    assert_eq!(placement_of(&summand), Partial);
    assert_eq!(
        placement_of(&Group::float_add(summand.clone(), summand)),
        Partial
    );
    assert_eq!(placement_of(&Group::float_sum_dim(x.clone(), 1)), Partial);
    assert_eq!(
        placement_of(&Group::float_reshape(x.clone(), Shape::new([4, 2, 3]))),
        Sharded { dim: 1 }
    );
    assert_eq!(
        placement_of(&Group::float_cat(vec![x.clone(), x.clone()], 0)),
        Sharded { dim: 1 }
    );
    assert_eq!(
        placement_of(&Group::float_slice(x.clone(), &[Slice::new(1, Some(3), 1)])),
        Sharded { dim: 1 }
    );
    assert_eq!(
        placement_of(&Group::float_expand(x, Shape::new([3, 4, 6]))),
        Sharded { dim: 2 }
    );
}

#[test]
fn integer_division_of_a_partial_sum_matches_one_device() {
    let device = group(2);
    let split = place(
        Group::int_from_data(TensorData::from([1i64, 1]), &device),
        Sharded { dim: 0 },
    );
    let quotient = Group::int_div_scalar(Group::int_sum(split), Scalar::Int(2));

    let quotient = block_on(Group::int_into_data(quotient)).expect("The tensor can be read");
    assert_eq!(quotient, TensorData::from([1i64]));
}

#[test]
#[should_panic(expected = "Cannot place")]
fn a_dim_shorter_than_the_group_is_not_split() {
    let device = group(3);
    place(
        Group::float_from_data(data([2, 6], 1), &device),
        Sharded { dim: 0 },
    );
}

#[test]
fn a_replayed_graph_runs_each_invocation_at_its_own_placements() {
    // (a + b) * b, the sum an intermediate the replay names itself.
    let relative = |id, status| TensorIr {
        id: TensorId::new(id),
        shape: Shape::new([0, 1]),
        status,
        dtype: DType::F32,
    };
    let (a, b) = (
        relative(0, TensorStatus::ReadOnly),
        relative(1, TensorStatus::ReadOnly),
    );
    let graph = vec![
        OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Add(BinaryOpIr {
                lhs: a,
                rhs: b.clone(),
                out: relative(2, TensorStatus::NotInit),
            }),
        ),
        OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Mul(BinaryOpIr {
                lhs: relative(2, TensorStatus::ReadWrite),
                rhs: b,
                out: relative(3, TensorStatus::NotInit),
            }),
        ),
    ];
    let invocations = [
        ([4, 6], Sharded { dim: 1 }, Replicated),
        ([3, 5], Partial, Sharded { dim: 1 }),
        ([4, 6], Sharded { dim: 0 }, Sharded { dim: 1 }),
    ];

    for ranks in [2, 3] {
        let device = group(ranks);
        for (invocation, (shape, a_placement, b_placement)) in invocations.into_iter().enumerate() {
            let (a, b) = (data(shape, 1), data(shape, 2));
            let lhs = placed(&a, &device, a_placement);
            let rhs = placed(&b, &device, b_placement);
            let client = lhs.client.clone();
            let out = client.create_empty_handle();
            let bindings = GraphBindings {
                tensors: vec![
                    (TensorId::new(0), lhs.id()),
                    (TensorId::new(1), rhs.id()),
                    (TensorId::new(3), out),
                ],
                shapes: shape.to_vec(),
                scalars: vec![],
                ranges: vec![],
            };
            match invocation {
                0 => client.register_and_execute_graph(GraphId(0), graph.clone(), bindings),
                _ => client.execute_graph(GraphId(0), bindings),
            }

            let flex = |data: TensorData| Flex::float_from_data(data, &FlexDevice);
            let expected = read::<Flex>(Flex::float_mul(
                Flex::float_add(flex(a), flex(b.clone())),
                flex(b),
            ));
            let actual = read::<Group>(RouterTensor::new(
                out,
                Shape::new(shape),
                DType::F32,
                client,
            ));
            let case = format!("{ranks} ranks, {a_placement:?} with {b_placement:?}");
            assert_close(actual, expected, &case);
        }
    }
}

#[test]
fn an_alias_keeps_its_source_value_and_placement_once_the_source_is_dropped() {
    let source_data = data([4, 6], 1);
    let source = place(
        Group::float_from_data(source_data.clone(), &group(2)),
        Sharded { dim: 1 },
    );
    let client = source.client.clone();
    let alias = client.create_empty_handle();
    client.register_alias(alias, source.id());
    let alias = RouterTensor::new(alias, source.shape(), source.dtype(), client);
    drop(source);

    assert_eq!(placement_of(&alias), Sharded { dim: 1 });
    assert_close(read::<Group>(alias), source_data, "alias");
}

struct Mlp;

impl Mlp {
    /// Leaves: x, w1, b1, w2, b2.
    fn run<B: AutodiffBackend>(leaves: [FloatTensor<B>; 5], target: FloatTensor<B>) -> Placed<B> {
        let [x, w1, b1, w2, b2] = leaves.clone();
        let hidden = B::relu(B::linear(x, w1, Some(b1)));
        let output = B::linear(hidden, w2, Some(b2));
        let diff = B::float_sub(output.clone(), target);
        let loss = B::float_mean(B::float_mul(diff.clone(), diff));
        Placed::backward(output, loss, &leaves)
    }
}

struct Attention;

impl Attention {
    const HEADS: usize = 4;
    const HEAD_DIM: usize = 3;
    const EMBED: usize = Self::HEADS * Self::HEAD_DIM;

    fn run<B: AutodiffBackend>(x: FloatTensor<B>, weights: [FloatTensor<B>; 4]) -> Placed<B> {
        let [wq, wk, wv, wo] = weights.clone();
        let shape = x.shape();
        let (batch, seq) = (shape[0], shape[1]);
        let heads = |weight: FloatTensor<B>| {
            let projected = B::linear(x.clone(), weight, None);
            let split = B::float_reshape(
                projected,
                Shape::new([batch, seq, Self::HEADS, Self::HEAD_DIM]),
            );
            B::float_swap_dims(split, 1, 2)
        };
        let (q, k, v) = (heads(wq), heads(wk), heads(wv));
        let scores = B::float_div_scalar(
            B::float_matmul(q, B::float_swap_dims(k, 2, 3)),
            Scalar::Float((Self::HEAD_DIM as f64).sqrt()),
        );
        let context = B::float_matmul(B::softmax(scores, 3), v);
        let merged = B::float_reshape(
            B::float_swap_dims(context, 1, 2),
            Shape::new([batch, seq, Self::EMBED]),
        );
        let output = B::float_add(B::linear(merged, wo, None), x.clone());
        let loss = B::float_mean(B::float_mul(output.clone(), output.clone()));
        let leaves: Vec<FloatTensor<B>> = [x].into_iter().chain(weights).collect();
        Placed::backward(output, loss, &leaves)
    }
}

fn embedding<B: AutodiffBackend>(
    weights: FloatTensor<B>,
    indices: IntTensor<B>,
    scale: FloatTensor<B>,
) -> Placed<B> {
    let output = B::embedding(weights.clone(), indices);
    let loss = B::float_mean(B::float_mul(output.clone(), scale));
    Placed::backward(output, loss, &[weights])
}

/// A forward output and each leaf's gradient, still on the backend.
struct Placed<B: AutodiffBackend> {
    output: FloatTensor<B::InnerBackend>,
    grads: Vec<FloatTensor<B::InnerBackend>>,
}

impl<B: AutodiffBackend> Placed<B> {
    fn backward(output: FloatTensor<B>, loss: FloatTensor<B>, leaves: &[FloatTensor<B>]) -> Self {
        let grads = B::backward(loss);
        Self {
            output: B::inner(output),
            grads: leaves
                .iter()
                .map(|leaf| B::grad(leaf, &grads).expect("Every leaf has a gradient"))
                .collect(),
        }
    }

    fn assert_matches<E: AutodiffBackend>(self, expected: Placed<E>, case: &str) {
        assert_close(
            read::<B::InnerBackend>(self.output),
            read::<E::InnerBackend>(expected.output),
            case,
        );
        for (actual, expected) in self.grads.into_iter().zip(expected.grads) {
            assert_close(
                read::<B::InnerBackend>(actual),
                read::<E::InnerBackend>(expected),
                case,
            );
        }
    }
}

fn assert_unary_matches(
    ranks: usize,
    input: &TensorData,
    on_group: impl Fn(FloatTensor<Group>) -> FloatTensor<Group>,
    on_flex: impl Fn(FloatTensor<Flex>) -> FloatTensor<Flex>,
) {
    let expected = read::<Flex>(on_flex(Flex::float_from_data(input.clone(), &FlexDevice)));
    let device = group(ranks);
    for placement in placements(input.shape(), ranks) {
        let actual = read::<Group>(on_group(placed(input, &device, placement)));
        assert_close(
            actual,
            expected.clone(),
            &format!("{ranks} ranks, {placement:?}"),
        );
    }
}

fn assert_binary_matches(
    ranks: usize,
    lhs: &TensorData,
    rhs: &TensorData,
    on_group: impl Fn(FloatTensor<Group>, FloatTensor<Group>) -> FloatTensor<Group>,
    on_flex: impl Fn(FloatTensor<Flex>, FloatTensor<Flex>) -> FloatTensor<Flex>,
) {
    let expected = read::<Flex>(on_flex(
        Flex::float_from_data(lhs.clone(), &FlexDevice),
        Flex::float_from_data(rhs.clone(), &FlexDevice),
    ));
    let device = group(ranks);
    for lhs_placement in placements(lhs.shape(), ranks) {
        for rhs_placement in placements(rhs.shape(), ranks) {
            let actual = read::<Group>(on_group(
                placed(lhs, &device, lhs_placement),
                placed(rhs, &device, rhs_placement),
            ));
            let case = format!("{ranks} ranks, {lhs_placement:?} with {rhs_placement:?}");
            assert_close(actual, expected.clone(), &case);
        }
    }
}

/// A partial sum is built as a contraction, `data @ I` with both split, so each rank holds a
/// different summand.
fn placed(
    data: &TensorData,
    device: &GroupDevice,
    placement: GroupPlacement,
) -> FloatTensor<Group> {
    let tensor = Group::float_from_data(data.clone(), device);
    if placement != Partial {
        return place(tensor, placement);
    }
    let num_dims = data.shape().num_dims();
    let len = data.shape()[num_dims - 1];
    let mut eye_shape = vec![1; num_dims - 2];
    eye_shape.extend([len, len]);
    let eye: Vec<f32> = (0..len * len)
        .map(|i| f32::from(u8::from(i / len == i % len)))
        .collect();
    let eye = Group::float_from_data(TensorData::new(eye, eye_shape), device);
    let partial = Group::float_matmul(
        place(tensor, Sharded { dim: num_dims - 1 }),
        place(eye, Sharded { dim: num_dims - 2 }),
    );
    assert_eq!(placement_of(&partial), Partial);
    partial
}

/// A partial sum needs a last dim to contract, at least as long as the group.
fn placements(shape: &Shape, ranks: usize) -> Vec<GroupPlacement> {
    let num_dims = shape.num_dims();
    let mut placements = vec![Replicated];
    if num_dims >= 2 && shape[num_dims - 1] >= ranks {
        placements.push(Partial);
    }
    placements.extend(
        (0..num_dims)
            .filter(|dim| shape[*dim] >= ranks)
            .map(|dim| Sharded { dim }),
    );
    placements
}

fn group(ranks: usize) -> GroupDevice {
    GroupDevice::new::<Flex>(&vec![FlexDevice; ranks])
}

fn place(tensor: FloatTensor<Group>, placement: GroupPlacement) -> FloatTensor<Group> {
    tensor.client.clone().place(tensor, placement)
}

fn leaf(data: TensorData) -> FloatTensor<Autodiff<Flex>> {
    Autodiff::<Flex>::float_set_require_grad(
        Autodiff::<Flex>::float_from_data(data, &FlexDevice),
        true,
    )
}

fn placed_leaf(
    data: &TensorData,
    device: &GroupDevice,
    placement: GroupPlacement,
) -> FloatTensor<Autodiff<Group>> {
    let placed = place(Group::float_from_data(data.clone(), device), placement);
    Autodiff::<Group>::float_set_require_grad(Autodiff::<Group>::from_inner(placed), true)
}

fn placement_of(tensor: &FloatTensor<Group>) -> GroupPlacement {
    tensor.client.placement(tensor)
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
