//! Tests that a reduce whose input is only read through computed coordinates (by `select` or
//! `gather`) in the preceding block computes correctly instead of being fused onto that block.

use super::*;
use burn_tensor::{Int, Tensor, TensorData, Tolerance, activation};

const ROWS: [[f32; 4]; 3] = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
];

/// A `select` keeping its input's shape, then a reduce of that input.
#[test]
fn select_then_reduce_of_its_input() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let x = TestTensor::<2>::from_data(ROWS, &device);
        let indices = TestTensorInt::<1>::from_data([2, 0, 1], &device);
        device.sync().unwrap();

        let selected = x.clone().select(0, indices);
        let sum = x.sum_dim(1);

        selected
            .into_data()
            .assert_eq(&TensorData::from([ROWS[2], ROWS[0], ROWS[1]]), false);
        sum.into_data()
            .assert_eq(&TensorData::from([[10.0], [26.0], [42.0]]), false);
    });
}

/// A `gather` with indices of its input's shape, then a reduce of that input.
#[test]
fn gather_then_reduce_of_its_input() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let x = TestTensor::<2>::from_data(ROWS, &device);
        let indices =
            TestTensorInt::<2>::from_data([[3, 2, 1, 0], [0, 1, 2, 3], [1, 1, 1, 1]], &device);
        device.sync().unwrap();

        let gathered = x.clone().gather(1, indices);
        let max = x.max_dim(1);

        gathered.into_data().assert_eq(
            &TensorData::from([
                [4.0, 3.0, 2.0, 1.0],
                [5.0, 6.0, 7.0, 8.0],
                [10.0, 10.0, 10.0, 10.0],
            ]),
            false,
        );
        max.into_data()
            .assert_eq(&TensorData::from([[4.0], [8.0], [12.0]]), false);
    });
}

/// Every reduce kind, over both axes, after a same-shape `select` along either axis, and the
/// reduce of the gather indices themselves.
#[test]
fn reduces_after_a_same_shape_index_match_the_values() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let values: Vec<f32> = (0..24).map(|i| ((i * 7) % 24) as f32 - 11.5).collect();
        let shape = [4, 6];
        let host = |row: usize, col: usize| values[row * shape[1] + col];

        for select_dim in 0..2 {
            for reduce_dim in 0..2 {
                for reduce in Reduce::ALL {
                    let x =
                        TestTensor::<2>::from_data(TensorData::new(values.clone(), shape), &device);
                    let len = shape[select_dim];
                    let order: Vec<i64> = (0..len as i64).rev().collect();
                    let indices =
                        TestTensorInt::<1>::from_data(TensorData::new(order, [len]), &device);
                    device.sync().unwrap();

                    let selected = x.clone().select(select_dim, indices);
                    let reduced = reduce.apply(x, reduce_dim);

                    let expected_selected: Vec<f32> = (0..shape[0])
                        .flat_map(|r| (0..shape[1]).map(move |c| (r, c)))
                        .map(|(r, c)| match select_dim {
                            0 => host(shape[0] - 1 - r, c),
                            _ => host(r, shape[1] - 1 - c),
                        })
                        .collect();
                    let context = format!("select dim {select_dim}, {reduce:?} dim {reduce_dim}");
                    assert_eq!(
                        selected.into_data().to_vec::<f32>().unwrap(),
                        expected_selected,
                        "{context}"
                    );
                    reduced.into_data().assert_approx_eq::<FloatElem>(
                        &reduce.expected(&values, shape, reduce_dim),
                        Tolerance::default(),
                    );
                }
            }
        }

        // The indices of a gather are an indexed input too.
        let x = TestTensor::<2>::from_data(ROWS, &device);
        let indices: Tensor<2, Int> =
            TestTensorInt::<2>::from_data([[3, 2, 1, 0], [0, 1, 2, 3], [1, 1, 1, 1]], &device);
        device.sync().unwrap();
        let gathered = x.gather(1, indices.clone());
        let index_sum = indices.sum_dim(1);
        gathered.into_data().assert_eq(
            &TensorData::from([
                [4.0, 3.0, 2.0, 1.0],
                [5.0, 6.0, 7.0, 8.0],
                [10.0, 10.0, 10.0, 10.0],
            ]),
            false,
        );
        index_sum
            .into_data()
            .assert_eq(&TensorData::from([[6], [6], [4]]), false);
    });
}

#[derive(Clone, Copy, Debug)]
enum Reduce {
    Sum,
    Mean,
    Max,
    Min,
    ArgMax,
    Softmax,
}

impl Reduce {
    const ALL: [Reduce; 6] = [
        Reduce::Sum,
        Reduce::Mean,
        Reduce::Max,
        Reduce::Min,
        Reduce::ArgMax,
        Reduce::Softmax,
    ];

    fn apply(self, x: TestTensor<2>, dim: usize) -> TestTensor<2> {
        match self {
            Reduce::Sum => x.sum_dim(dim),
            Reduce::Mean => x.mean_dim(dim),
            Reduce::Max => x.max_dim(dim),
            Reduce::Min => x.min_dim(dim),
            Reduce::ArgMax => x.argmax(dim).float(),
            Reduce::Softmax => activation::softmax(x, dim),
        }
    }

    /// The reduce over `dim` of the row-major `values` of `shape`, computed on the host.
    fn expected(self, values: &[f32], shape: [usize; 2], dim: usize) -> TensorData {
        let lanes = |i: usize| -> Vec<f32> {
            (0..shape[dim])
                .map(|j| match dim {
                    0 => values[j * shape[1] + i],
                    _ => values[i * shape[1] + j],
                })
                .collect()
        };
        let outer = shape[1 - dim];
        let mut out_shape = shape;
        if !matches!(self, Reduce::Softmax) {
            out_shape[dim] = 1;
        }
        let mut out = vec![0.0f32; out_shape[0] * out_shape[1]];
        for i in 0..outer {
            let lane = lanes(i);
            let max = lane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let reduced: Vec<f32> = match self {
                Reduce::Sum => vec![lane.iter().sum()],
                Reduce::Mean => vec![lane.iter().sum::<f32>() / lane.len() as f32],
                Reduce::Max => vec![max],
                Reduce::Min => vec![lane.iter().copied().fold(f32::INFINITY, f32::min)],
                Reduce::ArgMax => vec![lane.iter().position(|&v| v == max).unwrap() as f32],
                Reduce::Softmax => {
                    let exps: Vec<f32> = lane.iter().map(|v| (v - max).exp()).collect();
                    let total: f32 = exps.iter().sum();
                    exps.iter().map(|e| e / total).collect()
                }
            };
            for (j, value) in reduced.into_iter().enumerate() {
                let index = match dim {
                    0 => j * out_shape[1] + i,
                    _ => i * out_shape[1] + j,
                };
                out[index] = value;
            }
        }
        TensorData::new(out, out_shape)
    }
}
