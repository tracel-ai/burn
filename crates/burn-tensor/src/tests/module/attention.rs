#[burn_tensor_testgen::testgen(module_attention)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;
    use burn_tensor::module::attention;
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_attention_basic() {
        // Use batch_size=1, num_heads=1, seq_len=2, head_dim=2 for simplicity
        let query = TestTensor::<4>::from([[[[1.0, 0.0], [0.0, 1.0]]]]); // shape: [1, 1, 2, 2]
        let key = TestTensor::<4>::from([[[[1.0, 0.0], [0.0, 1.0]]]]); // shape: [1, 1, 2, 2]
        let value = TestTensor::<4>::from([[[[1.0, 2.0], [3.0, 4.0]]]]); // shape: [1, 1, 2, 2]

        // No mask
        let output = attention(query, key, value, None);

        let expected = TestTensor::<4>::from([[[[1.6605, 2.6605], [2.3395, 3.3395]]]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::relative(1e-3));
    }

    #[test]
    fn test_attention_with_mask() {
        let query = TestTensor::<4>::from([[[[1.0, 0.0], [0.0, 1.0]]]]);
        let key = TestTensor::<4>::from([[[[1.0, 0.0], [0.0, 1.0]]]]);
        let value = TestTensor::<4>::from([[[[1.0, 2.0], [3.0, 4.0]]]]);

        // Mask out the second key (index 1)
        let mask = TestTensorBool::<4>::from([[[[false, true], [false, true]]]]);

        let output = attention(query, key, value, Some(mask));

        let expected = TestTensor::<4>::from([[[[1.0, 2.0], [1.0, 2.0]]]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    fn test_attention_larger_shape() {
        // Shape: [batch=2, heads=2, seq=3, dim=2]
        let query = TestTensor::<4>::from([
            [
                // batch 0
                [[1.0, 0.0], [0.0, 1.0]], // token 0: 2 heads
                [[1.0, 0.0], [0.0, 1.0]], // token 1
                [[1.0, 0.0], [0.0, 1.0]], // token 2
            ],
            [
                // batch 1
                [[1.0, 1.0], [0.5, 0.5]],
                [[1.0, 1.0], [0.5, 0.5]],
                [[1.0, 1.0], [0.5, 0.5]],
            ],
        ]);

        let key = query.clone(); // symmetric test
        let value = TestTensor::<4>::from([
            [
                // batch 0
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            ],
            [
                // batch 1
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            ],
        ]);

        // Mask out key 2 for every query in every batch/head
        // Shape: [2, 2, 3, 3] → batch, head, seq_q, seq_k
        let mask = TestTensorBool::<4>::from([
            [
                // batch 0
                [
                    [false, false, true],
                    [false, false, true],
                    [false, false, true],
                ],
                [
                    [false, false, true],
                    [false, false, true],
                    [false, false, true],
                ],
            ],
            [
                // batch 1
                [
                    [false, false, true],
                    [false, false, true],
                    [false, false, true],
                ],
                [
                    [false, false, true],
                    [false, false, true],
                    [false, false, true],
                ],
            ],
        ]);

        let output = attention(query, key, value, Some(mask));

        // We won't compute exact expected values here — instead, just check shape
        let shape = output.shape().dims();
        assert_eq!(shape, [2, 2, 3, 2]); // [batch, heads, seq_len, head_dim]
    }
}
