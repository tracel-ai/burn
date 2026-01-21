// Import the shared macro
use crate::include_models;
include_models!(space_to_depth);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn space_to_depth() {
        let device = Default::default();
        let model: space_to_depth::Model<TestBackend> = space_to_depth::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats(
            [
                [[
                    [0.5, -0.14, 0.65, 1.52, -0.23, -0.23],
                    [1.58, 0.77, -0.47, 0.54, -0.46, -0.47],
                    [0.24, -1.91, -1.72, -0.56, -1.01, 0.31],
                    [-0.91, -1.41, 1.47, -0.23, 0.07, -1.42],
                ]],
                [[
                    [-0.54, 0.11, -1.15, 0.38, -0.6, -0.29],
                    [-0.6, 1.85, -0.01, -1.06, 0.82, -1.22],
                    [0.21, -1.96, -1.33, 0.2, 0.74, 0.17],
                    [-0.12, -0.3, -1.48, -0.72, -0.46, 1.06],
                ]],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [
                [[0.5, 0.65, -0.23], [0.24, -1.72, -1.01]],
                [[-0.14, 1.52, -0.23], [-1.91, -0.56, 0.31]],
                [[1.58, -0.47, -0.46], [-0.91, 1.47, 0.07]],
                [[0.77, 0.54, -0.47], [-1.41, -0.23, -1.42]],
            ],
            [
                [[-0.54, -1.15, -0.6], [0.21, -1.33, 0.74]],
                [[0.11, 0.38, -0.29], [-1.96, 0.2, 0.17]],
                [[-0.6, -0.01, 0.82], [-0.12, -1.48, -0.46]],
                [[1.85, -1.06, -1.22], [-0.3, -0.72, 1.06]],
            ],
        ]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
