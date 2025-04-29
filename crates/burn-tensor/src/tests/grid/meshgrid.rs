#[burn_tensor_testgen::testgen(meshgrid)]
mod tests {
    use super::*;
    use burn_tensor::BasicOps;
    use burn_tensor::backend::Backend;
    use burn_tensor::grid::{GridIndexing, GridSparsity, MeshGridOptions, meshgrid};
    use burn_tensor::{Int, Shape, Tensor, TensorData};

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
            &meshgrid(
                &[x.clone(), y.clone(), z.clone()],
                MeshGridOptions::default(),
            ),
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
                MeshGridOptions {
                    sparsity: GridSparsity::Sparse,
                    indexing: GridIndexing::Matrix,
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
                MeshGridOptions::new(GridSparsity::Sparse, GridIndexing::Cartesian),
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
                MeshGridOptions {
                    sparsity: GridSparsity::Sparse,
                    indexing: GridIndexing::Cartesian,
                },
            ),
            &[
                x.clone().reshape([4, 1, 1]).swap_dims(0, 1),
                y.clone().reshape([1, 2, 1]).swap_dims(0, 1),
                z.clone().reshape([1, 1, 2]).swap_dims(0, 1),
            ],
        );
    }
}
