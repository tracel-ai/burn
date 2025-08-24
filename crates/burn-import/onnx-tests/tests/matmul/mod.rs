// Import the shared macro
use crate::include_models;
include_models!(matmul);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn matmul() {
        // Initialize the model with weights (loaded from the exported file)
        let model: matmul::Model<TestBackend> = matmul::Model::default();

        let device = Default::default();
        let a = Tensor::<TestBackend, 1, Int>::arange(0..24, &device)
            .reshape([1, 2, 3, 4])
            .float();
        let b = Tensor::<TestBackend, 1, Int>::arange(0..16, &device)
            .reshape([1, 2, 4, 2])
            .float();
        let c = Tensor::<TestBackend, 1, Int>::arange(0..96, &device)
            .reshape([2, 3, 4, 4])
            .float();
        let d = Tensor::<TestBackend, 1, Int>::arange(0..4, &device).float();

        let (output_mm, output_mv, output_vm) = model.forward(a, b, c, d);
        // matrix-matrix `a @ b`
        let expected_mm = TensorData::from([[
            [[28f32, 34.], [76., 98.], [124., 162.]],
            [[604., 658.], [780., 850.], [956., 1042.]],
        ]]);
        // matrix-vector `c @ d` where the lhs vector is expanded and broadcasted to the correct dims
        let expected_mv = TensorData::from([
            [
                [14f32, 38., 62., 86.],
                [110., 134., 158., 182.],
                [206., 230., 254., 278.],
            ],
            [
                [302., 326., 350., 374.],
                [398., 422., 446., 470.],
                [494., 518., 542., 566.],
            ],
        ]);
        // vector-matrix `d @ c` where the rhs vector is expanded and broadcasted to the correct dims
        let expected_vm = TensorData::from([
            [
                [56f32, 62., 68., 74.],
                [152., 158., 164., 170.],
                [248., 254., 260., 266.],
            ],
            [
                [344., 350., 356., 362.],
                [440., 446., 452., 458.],
                [536., 542., 548., 554.],
            ],
        ]);

        output_mm.to_data().assert_eq(&expected_mm, true);
        output_vm.to_data().assert_eq(&expected_vm, true);
        output_mv.to_data().assert_eq(&expected_mv, true);
    }
}
