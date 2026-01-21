// Import the shared macro
use crate::include_models;
include_models!(instance_norm1d, instance_norm2d, instance_norm3d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn instance_norm1d() {
        let device = Default::default();
        let model: instance_norm1d::Model<TestBackend> = instance_norm1d::Model::default();

        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [0., 1., 2., 3.], //
                    [4., 5., 6., 7.],
                    [8., 9., 10., 11.],
                ],
                [
                    [12., 13., 14., 15.], //
                    [16., 17., 18., 19.],
                    [20., 21., 22., 23.],
                ],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [
                [-1.3416f32, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
            ],
            [
                [-1.3416, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
            ],
        ]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn instance_norm2d() {
        let device = Default::default();
        let model: instance_norm2d::Model<TestBackend> = instance_norm2d::Model::default();

        let input = Tensor::<TestBackend, 4>::from_floats(
            [
                [
                    [
                        [0., 1., 2., 3.], //
                        [4., 5., 6., 7.],
                        [8., 9., 10., 11.],
                    ],
                    [
                        [12., 13., 14., 15.], //
                        [16., 17., 18., 19.],
                        [20., 21., 22., 23.],
                    ],
                ],
                [
                    [
                        [24., 25., 26., 27.], //
                        [28., 29., 30., 31.],
                        [32., 33., 34., 35.],
                    ],
                    [
                        [36., 37., 38., 39.], //
                        [40., 41., 42., 43.],
                        [44., 45., 46., 47.],
                    ],
                ],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [
                [
                    [-1.5933, -1.3036, -1.0139, -0.7242],
                    [-0.4345, -0.1448, 0.1448, 0.4345],
                    [0.7242, 1.0139, 1.3036, 1.5933],
                ],
                [
                    [-1.5933, -1.3036, -1.0139, -0.7242],
                    [-0.4345, -0.1448, 0.1448, 0.4345],
                    [0.7242, 1.0139, 1.3036, 1.5933],
                ],
            ],
            [
                [
                    [-1.5933, -1.3036, -1.0139, -0.7242],
                    [-0.4345, -0.1448, 0.1448, 0.4345],
                    [0.7242, 1.0139, 1.3036, 1.5933],
                ],
                [
                    [-1.5933, -1.3036, -1.0139, -0.7242],
                    [-0.4345, -0.1448, 0.1448, 0.4345],
                    [0.7242, 1.0139, 1.3036, 1.5933],
                ],
            ],
        ]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn instance_norm3d() {
        let device = Default::default();
        let model: instance_norm3d::Model<TestBackend> = instance_norm3d::Model::default();

        let input = Tensor::<TestBackend, 5>::from_floats(
            [
                [
                    [[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
                    [[[8., 9.], [10., 11.]], [[12., 13.], [14., 15.]]],
                    [[[16., 17.], [18., 19.]], [[20., 21.], [22., 23.]]],
                ],
                [
                    [[[24., 25.], [26., 27.]], [[28., 29.], [30., 31.]]],
                    [[[32., 33.], [34., 35.]], [[36., 37.], [38., 39.]]],
                    [[[40., 41.], [42., 43.]], [[44., 45.], [46., 47.]]],
                ],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [
                [
                    [[-1.5275, -1.0911], [-0.6547, -0.2182]], //
                    [[0.2182, 0.6547], [1.0911, 1.5275]],
                ],
                [
                    [[-1.5275, -1.0911], [-0.6547, -0.2182]], //
                    [[0.2182, 0.6547], [1.0911, 1.5275]],
                ],
                [
                    [[-1.5275, -1.0911], [-0.6547, -0.2182]], //
                    [[0.2182, 0.6547], [1.0911, 1.5275]],
                ],
            ],
            [
                [
                    [[-1.5275, -1.0911], [-0.6547, -0.2182]], //
                    [[0.2182, 0.6547], [1.0911, 1.5275]],
                ],
                [
                    [[-1.5275, -1.0911], [-0.6547, -0.2182]], //
                    [[0.2182, 0.6547], [1.0911, 1.5275]],
                ],
                [
                    [[-1.5275, -1.0911], [-0.6547, -0.2182]], //
                    [[0.2182, 0.6547], [1.0911, 1.5275]],
                ],
            ],
        ]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
