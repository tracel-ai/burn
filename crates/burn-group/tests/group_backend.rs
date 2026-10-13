use burn_backend::{
    Backend, Scalar, Shape, TensorData, TensorMetadata,
    ops::{ActivationOps, FloatTensorOps},
    tensor::FloatTensor,
};
use burn_flex::{Flex, FlexDevice};
use burn_group::{
    GroupBackend, GroupDevice,
    GroupPlacement::{self, Partial, Replicated, Sharded},
};
use burn_std::future::block_on;

type Group = GroupBackend;

const MEMBERS: [usize; 4] = [1, 2, 3, 4];

macro_rules! check_unary {
    ($members:expr, $input:expr, |$backend:ident, $a:ident| $body:expr) => {
        check_unary!($members, $input, [] |$backend, $a| $body)
    };
    ($members:expr, $input:expr, [$($env:ident: $ty:ty),*] |$backend:ident, $a:ident| $body:expr) => {{
        fn op<$backend: Backend>($a: FloatTensor<$backend>, $($env: $ty),*) -> FloatTensor<$backend> {
            $body
        }
        assert_unary_matches(
            $members,
            $input,
            |a| op::<Group>(a, $($env.clone()),*),
            |a| op::<Flex>(a, $($env.clone()),*),
        )
    }};
}

macro_rules! check_binary {
    ($members:expr, $lhs:expr, $rhs:expr, |$backend:ident, $a:ident, $b:ident| $body:expr) => {
        check_binary!($members, $lhs, $rhs, [] |$backend, $a, $b| $body)
    };
    ($members:expr, $lhs:expr, $rhs:expr, [$($env:ident: $ty:ty),*] |$backend:ident, $a:ident, $b:ident| $body:expr) => {{
        fn op<$backend: Backend>(
            $a: FloatTensor<$backend>,
            $b: FloatTensor<$backend>,
            $($env: $ty),*
        ) -> FloatTensor<$backend> {
            $body
        }
        assert_binary_matches(
            $members,
            $lhs,
            $rhs,
            |a, b| op::<Group>(a, b, $($env.clone()),*),
            |a, b| op::<Flex>(a, b, $($env.clone()),*),
        )
    }};
}

#[test]
fn elementwise_ops_match_one_device_for_every_placement() {
    for members in MEMBERS {
        for rhs in [[4, 6], [1, 6], [4, 1]] {
            let (lhs, rhs) = (data([4, 6], 1), data(rhs, 2));
            check_binary!(members, &lhs, &rhs, |B, a, b| B::float_add(a, b));
            check_binary!(members, &lhs, &rhs, |B, a, b| B::float_sub(a, b));
            check_binary!(members, &lhs, &rhs, |B, a, b| B::float_mul(a, b));
            check_binary!(members, &lhs, &positive(rhs), |B, a, b| B::float_div(a, b));
        }
        let input = data([4, 6], 3);
        check_unary!(members, &input, |B, a| B::float_neg(a));
        check_unary!(members, &input, |B, a| B::float_mul_scalar(
            a,
            Scalar::Float(3.0)
        ));
        check_unary!(members, &input, |B, a| B::float_div_scalar(
            a,
            Scalar::Float(3.0)
        ));
        check_unary!(members, &input, |B, a| B::relu(a));
        check_unary!(members, &input, |B, a| B::gelu(a));
    }
}

#[test]
fn matmuls_match_one_device_for_every_placement() {
    for members in MEMBERS {
        for (lhs, rhs) in [
            (vec![4, 6], vec![6, 5]),
            (vec![3, 4, 6], vec![3, 6, 5]),
            (vec![3, 4, 6], vec![1, 6, 5]),
        ] {
            let (lhs, rhs) = (data(lhs, 1), data(rhs, 2));
            check_binary!(members, &lhs, &rhs, |B, a, b| B::float_matmul(a, b));
        }
        let (x, weight, bias) = (data([2, 4, 6], 1), data([6, 5], 2), data([5], 3));
        check_binary!(members, &x, &weight, |B, x, weight| B::linear(
            x, weight, None
        ));
        check_binary!(members, &x, &weight, [bias: TensorData] |B, x, weight| {
            let device = x.device();
            B::linear(x, weight, Some(B::float_from_data(bias, &device)))
        });
    }
}

#[test]
fn sums_match_one_device_for_every_placement() {
    for members in MEMBERS {
        let input = data([4, 3, 5], 1);
        for dim in 0..3 {
            check_unary!(members, &input, [dim: usize] |B, a| B::float_sum_dim(a, dim));
        }
        check_unary!(members, &input, |B, a| B::float_sum(a));
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
    assert_eq!(placement_of(&Group::relu(x)), Sharded { dim: 1 });
}

#[test]
#[should_panic(expected = "No collective places")]
fn a_dim_shorter_than_the_group_is_not_split() {
    let device = group(3);
    place(
        Group::float_from_data(data([2, 6], 1), &device),
        Sharded { dim: 0 },
    );
}

#[test]
fn every_placement_moves_to_every_other_and_keeps_its_value() {
    let input = data([4, 6], 1);
    for members in MEMBERS {
        let group = group(members);
        for from in placements(input.shape(), members) {
            for to in placements(input.shape(), members) {
                if to == Partial && from != Partial {
                    continue;
                }
                let moved = place(placed(&input, &group, from), to);
                assert_eq!(placement_of(&moved), to);
                let case = format!("{members} members, {from:?} to {to:?}");
                assert_close(read::<Group>(moved), input.clone(), &case);
            }
        }
    }
}

fn assert_unary_matches(
    members: usize,
    input: &TensorData,
    on_group: impl Fn(FloatTensor<Group>) -> FloatTensor<Group>,
    on_flex: impl Fn(FloatTensor<Flex>) -> FloatTensor<Flex>,
) {
    let expected = read::<Flex>(on_flex(Flex::float_from_data(input.clone(), &FlexDevice)));
    let device = group(members);
    for placement in placements(input.shape(), members) {
        let actual = read::<Group>(on_group(placed(input, &device, placement)));
        assert_close(
            actual,
            expected.clone(),
            &format!("{members} members, {placement:?}"),
        );
    }
}

fn assert_binary_matches(
    members: usize,
    lhs: &TensorData,
    rhs: &TensorData,
    on_group: impl Fn(FloatTensor<Group>, FloatTensor<Group>) -> FloatTensor<Group>,
    on_flex: impl Fn(FloatTensor<Flex>, FloatTensor<Flex>) -> FloatTensor<Flex>,
) {
    let expected = read::<Flex>(on_flex(
        Flex::float_from_data(lhs.clone(), &FlexDevice),
        Flex::float_from_data(rhs.clone(), &FlexDevice),
    ));
    let device = group(members);
    for lhs_placement in placements(lhs.shape(), members) {
        for rhs_placement in placements(rhs.shape(), members) {
            let actual = read::<Group>(on_group(
                placed(lhs, &device, lhs_placement),
                placed(rhs, &device, rhs_placement),
            ));
            let case = format!("{members} members, {lhs_placement:?} with {rhs_placement:?}");
            assert_close(actual, expected.clone(), &case);
        }
    }
}

/// A partial sum is built as a contraction, `data @ I` with both split, so each member holds a
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
fn placements(shape: &Shape, members: usize) -> Vec<GroupPlacement> {
    let num_dims = shape.num_dims();
    let mut placements = vec![Replicated];
    if num_dims >= 2 && shape[num_dims - 1] >= members {
        placements.push(Partial);
    }
    placements.extend(
        (0..num_dims)
            .filter(|dim| shape[*dim] >= members)
            .map(|dim| Sharded { dim }),
    );
    placements
}

fn group(members: usize) -> GroupDevice {
    GroupDevice::new::<Flex>(&vec![FlexDevice; members])
}

fn place(tensor: FloatTensor<Group>, placement: GroupPlacement) -> FloatTensor<Group> {
    tensor.client.clone().place(tensor, placement)
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
