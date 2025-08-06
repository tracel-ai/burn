#[burn_tensor_testgen::testgen(glu)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData, activation};

    #[test]
    fn test_glu_d3() {
        let tensor = TestTensor::<3>::from([[
            [
                -0.5710, -1.3416, 1.9128, -0.8257, -0.1331, -1.4804, -0.6281, -0.6115,
            ],
            [
                0.0267, -1.3834, 0.2752, 0.7844, -0.3549, -0.4274, 0.3290, -0.5459,
            ],
            [
                -1.6347, -2.0908, 1.8801, 0.3541, 0.2237, 1.0377, 2.4850, 0.3490,
            ],
        ]]);

        let output = activation::glu(tensor, 2);

        output.into_data().assert_eq(
            &TensorData::from([[
                [-0.2665, -0.2487, 0.6656, -0.2904],
                [0.0110, -0.5461, 0.1601, 0.2877],
                [-0.9084, -1.5439, 1.7355, 0.2077],
            ]]),
            false,
        );
    }
}
