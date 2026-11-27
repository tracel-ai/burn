//! Tests that a reduce whose input the preceding fused operations only read through computed
//! coordinates (`select`, `gather`) runs correctly instead of panicking the fusion server.

use super::*;
use burn_tensor::{TensorData, Tolerance, activation};

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

/// A `gather` with indices of its input's shape, then a reduce of that input and of the indices,
/// which are an indexed input too.
#[test]
fn gather_then_reduce_of_its_inputs() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let x = TestTensor::<2>::from_data(ROWS, &device);
        let indices =
            TestTensorInt::<2>::from_data([[3, 2, 1, 0], [0, 1, 2, 3], [1, 1, 1, 1]], &device);
        device.sync().unwrap();

        let gathered = x.clone().gather(1, indices.clone());
        let max = x.max_dim(1);
        let index_sum = indices.sum_dim(1);

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
        index_sum
            .into_data()
            .assert_eq(&TensorData::from([[6], [6], [4]]), false);
    });
}

/// Over shapes, the `select` axis and the reduce kind and axis: the same operations give the same
/// values whether they are queued together or split by a sync between the `select` and the reduce.
#[test]
fn select_and_reduce_match_the_split_run() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let reduces: [(&str, fn(TestTensor<3>, usize) -> TestTensor<3>); 4] = [
            ("sum_dim", |x, dim| x.sum_dim(dim)),
            ("mean_dim", |x, dim| x.mean_dim(dim)),
            ("argmax", |x, dim| x.argmax(dim).float()),
            ("softmax", |x, dim| activation::softmax(x, dim)),
        ];

        for shape in [[2, 3, 4], [4, 1, 5], [3, 5, 2]] {
            let n: usize = shape.iter().product();
            // Distinct values, so argmax has no ties: 7 is coprime with every `n` above.
            let values: Vec<f32> = (0..n).map(|i| ((i * 7) % n) as f32 / n as f32).collect();
            for select_dim in 0..3 {
                for reduce_dim in 0..3 {
                    for (name, reduce) in reduces {
                        let run = |split: bool| {
                            let x = TestTensor::<3>::from_data(
                                TensorData::new(values.clone(), shape),
                                &device,
                            );
                            let len = shape[select_dim];
                            let order: Vec<i64> = (0..len as i64).rev().collect();
                            let indices = TestTensorInt::<1>::from_data(
                                TensorData::new(order, [len]),
                                &device,
                            );
                            device.sync().unwrap();

                            let selected = x.clone().select(select_dim, indices);
                            if split {
                                device.sync().unwrap();
                            }
                            let reduced = reduce(x, reduce_dim);
                            (selected.into_data(), reduced.into_data())
                        };
                        let (selected, reduced) = run(false);
                        let (selected_split, reduced_split) = run(true);
                        let case =
                            format!("{shape:?}: select dim {select_dim}, {name} dim {reduce_dim}");
                        assert_eq!(selected, selected_split, "{case}");
                        reduced.assert_approx_eq::<FloatElem>(&reduced_split, Tolerance::default());
                    }
                }
            }
        }
    });
}
