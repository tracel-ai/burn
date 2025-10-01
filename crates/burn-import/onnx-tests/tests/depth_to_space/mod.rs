// Import the shared macro
use crate::include_models;
include_models!(depth_to_space_dcr, depth_to_space_crd);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn depth_to_space_dcr() {
        let device = Default::default();
        let model: depth_to_space_dcr::Model<TestBackend> = depth_to_space_dcr::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats(
            [
                [
                    [[0.5, -0.14, 0.65], [1.52, -0.23, -0.23]],
                    [[1.58, 0.77, -0.47], [0.54, -0.46, -0.47]],
                    [[0.24, -1.91, -1.72], [-0.56, -1.01, 0.31]],
                    [[-0.91, -1.41, 1.47], [-0.23, 0.07, -1.42]],
                ],
                [
                    [[-0.54, 0.11, -1.15], [0.38, -0.6, -0.29]],
                    [[-0.6, 1.85, -0.01], [-1.06, 0.82, -1.22]],
                    [[0.21, -1.96, -1.33], [0.2, 0.74, 0.17]],
                    [[-0.12, -0.3, -1.48], [-0.72, -0.46, 1.06]],
                ],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [[
                [0.5, 1.58, -0.14, 0.77, 0.65, -0.47],
                [0.24, -0.91, -1.91, -1.41, -1.72, 1.47],
                [1.52, 0.54, -0.23, -0.46, -0.23, -0.47],
                [-0.56, -0.23, -1.01, 0.07, 0.31, -1.42],
            ]],
            [[
                [-0.54, -0.6, 0.11, 1.85, -1.15, -0.01],
                [0.21, -0.12, -1.96, -0.3, -1.33, -1.48],
                [0.38, -1.06, -0.6, 0.82, -0.29, -1.22],
                [0.2, -0.72, 0.74, -0.46, 0.17, 1.06],
            ]],
        ]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn depth_to_space_crd() {
        let device = Default::default();
        let model: depth_to_space_crd::Model<TestBackend> = depth_to_space_crd::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats(
            [
                [
                    [[0.34, -1.76, 0.32], [-0.39, -0.68, 0.61]],
                    [[1.03, 0.93, -0.84], [-0.31, 0.33, 0.98]],
                    [[-0.48, -0.19, -1.11], [-1.2, 0.81, 1.36]],
                    [[-0.07, 1., 0.36], [-0.65, 0.36, 1.54]],
                ],
                [
                    [[-0.04, 1.56, -2.62], [0.82, 0.09, -0.3]],
                    [[0.09, -1.99, -0.22], [0.36, 1.48, -0.52]],
                    [[-0.81, -0.5, 0.92], [0.33, -0.53, 0.51]],
                    [[0.1, 0.97, -0.7], [-0.33, -0.39, -1.46]],
                ],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [[
                [0.34, 1.03, -1.76, 0.93, 0.32, -0.84],
                [-0.48, -0.07, -0.19, 1., -1.11, 0.36],
                [-0.39, -0.31, -0.68, 0.33, 0.61, 0.98],
                [-1.2, -0.65, 0.81, 0.36, 1.36, 1.54],
            ]],
            [[
                [-0.04, 0.09, 1.56, -1.99, -2.62, -0.22],
                [-0.81, 0.1, -0.5, 0.97, 0.92, -0.7],
                [0.82, 0.36, 0.09, 1.48, -0.3, -0.52],
                [0.33, -0.33, -0.53, -0.39, 0.51, -1.46],
            ]],
        ]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
