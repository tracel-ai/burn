use crate::*;
use burn_tensor::BasicOps;
use burn_tensor::Tensor;
use burn_tensor::TensorData;
use burn_tensor::backend::Backend;
use burn_tensor::grid::{
    GridIndexing, GridOptions, GridSparsity, IndexPos, meshgrid, meshgrid_stack,
};

fn assert_tensors_equal<const N: usize, B: Backend, K>(
    actual: &[Tensor<B, N, K>; N],
    expected: &[Tensor<B, N, K>; N],
) where
    K: BasicOps<B>,
{
    for (a, e) in actual.iter().zip(expected.iter()) {
        a.clone()
            .into_data()
            .assert_eq(&e.clone().into_data(), true);
    }
}

#[test]
fn test_meshgrid() {
    let x = TestTensor::<1>::from([1, 2, 3, 4]);
    let y = TestTensor::<1>::from([5, 6]);
    let z = TestTensor::<1>::from([7, 8]);

    let grid_shape = [x.dims()[0], y.dims()[0], z.dims()[0]];

    // 3D, Dense, Matrix
    assert_tensors_equal(
        &meshgrid(&[x.clone(), y.clone(), z.clone()], GridOptions::default()),
        &[
            x.clone().reshape([4, 1, 1]).expand(grid_shape),
            y.clone().reshape([1, 2, 1]).expand(grid_shape),
            z.clone().reshape([1, 1, 2]).expand(grid_shape),
        ],
    );
    assert_tensors_equal(
        &meshgrid(&[x.clone(), y.clone(), z.clone()], GridSparsity::Dense),
        &[
            x.clone().reshape([4, 1, 1]).expand(grid_shape),
            y.clone().reshape([1, 2, 1]).expand(grid_shape),
            z.clone().reshape([1, 1, 2]).expand(grid_shape),
        ],
    );
    assert_tensors_equal(
        &meshgrid(&[x.clone(), y.clone(), z.clone()], GridIndexing::Matrix),
        &[
            x.clone().reshape([4, 1, 1]).expand(grid_shape),
            y.clone().reshape([1, 2, 1]).expand(grid_shape),
            z.clone().reshape([1, 1, 2]).expand(grid_shape),
        ],
    );

    // 3D, Sparse, Matrix
    assert_tensors_equal(
        &meshgrid(
            &[x.clone(), y.clone(), z.clone()],
            GridOptions {
                indexing: GridIndexing::Matrix,
                sparsity: GridSparsity::Sparse,
            },
        ),
        &[
            x.clone().reshape([4, 1, 1]),
            y.clone().reshape([1, 2, 1]),
            z.clone().reshape([1, 1, 2]),
        ],
    );
    assert_tensors_equal(
        &meshgrid(&[x.clone(), y.clone(), z.clone()], GridSparsity::Sparse),
        &[
            x.clone().reshape([4, 1, 1]),
            y.clone().reshape([1, 2, 1]),
            z.clone().reshape([1, 1, 2]),
        ],
    );

    // 3D, Dense, Cartesian
    assert_tensors_equal(
        &meshgrid(&[x.clone(), y.clone(), z.clone()], GridIndexing::Cartesian),
        &[
            x.clone()
                .reshape([4, 1, 1])
                .expand(grid_shape)
                .swap_dims(0, 1),
            y.clone()
                .reshape([1, 2, 1])
                .expand(grid_shape)
                .swap_dims(0, 1),
            z.clone()
                .reshape([1, 1, 2])
                .expand(grid_shape)
                .swap_dims(0, 1),
        ],
    );

    // 3D, Sparse, Cartesian
    assert_tensors_equal(
        &meshgrid(
            &[x.clone(), y.clone(), z.clone()],
            GridOptions::new(GridIndexing::Cartesian, GridSparsity::Sparse),
        ),
        &[
            x.clone().reshape([4, 1, 1]).swap_dims(0, 1),
            y.clone().reshape([1, 2, 1]).swap_dims(0, 1),
            z.clone().reshape([1, 1, 2]).swap_dims(0, 1),
        ],
    );
    assert_tensors_equal(
        &meshgrid(
            &[x.clone(), y.clone(), z.clone()],
            GridOptions {
                indexing: GridIndexing::Cartesian,
                sparsity: GridSparsity::Sparse,
            },
        ),
        &[
            x.clone().reshape([4, 1, 1]).swap_dims(0, 1),
            y.clone().reshape([1, 2, 1]).swap_dims(0, 1),
            z.clone().reshape([1, 1, 2]).swap_dims(0, 1),
        ],
    );
}

#[test]
fn test_meshgrid_stack() {
    let tensors = [
        TestTensor::from([0.5, 1.0, 2.5]),
        TestTensor::from([0.5, 1.0]),
    ];

    let result: Tensor<_, 3> = meshgrid_stack(&tensors, IndexPos::First);
    result.to_data().assert_eq(
        &TensorData::from([
            [[0.5, 0.5], [1.0, 1.0], [2.5, 2.5]],
            [[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]],
        ]),
        false,
    );

    let result: Tensor<_, 3> = meshgrid_stack(&tensors, IndexPos::Last);
    result.to_data().assert_eq(
        &TensorData::from([
            [[0.5, 0.5], [0.5, 1.0]],
            [[1.0, 0.5], [1.0, 1.0]],
            [[2.5, 0.5], [2.5, 1.0]],
        ]),
        false,
    );
}
