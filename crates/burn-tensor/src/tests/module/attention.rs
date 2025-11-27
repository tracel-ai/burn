#[burn_tensor_testgen::testgen(module_attention)]
mod tests {
    use super::*;
    use burn_tensor::Distribution;
    use burn_tensor::TensorData;
    use burn_tensor::module::attention;
    use burn_tensor::module::naive_attention;
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_attention_no_mask() {
        let num_batches = 1;
        let num_heads = 1;
        let seq_q = 128;
        let seq_kv = 128;
        let head_dim = 64;
        let val_dim = 64;

        let query = TestTensor::<4>::random(
            [num_batches, num_heads, seq_q, head_dim],
            Distribution::Uniform(0., 1.),
            &Default::default(),
        );
        let key = TestTensor::<4>::random(
            [num_batches, num_heads, seq_kv, head_dim],
            Distribution::Uniform(0., 1.),
            &Default::default(),
        );
        let value = TestTensor::<4>::random(
            [num_batches, num_heads, seq_kv, val_dim],
            Distribution::Uniform(0., 1.),
            &Default::default(),
        );

        let output = attention(query.clone(), key.clone(), value.clone(), None);

        let expected = naive_attention::<TestBackend>(query, key, value, None);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::relative(1e-3));
    }

    #[test]
    fn test_attention_shape() {
        // Shape: [batch=2, heads=3, seq_q=3, head_dim=2]
        let query = TestTensor::<4>::from([
            [
                // batch 0
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], // token 0,1,2 per head
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            ],
            [
                // batch 1
                [[1.0, 1.0], [0.5, 0.5], [0.0, 1.0]],
                [[1.0, 1.0], [0.5, 0.5], [0.0, 1.0]],
                [[1.0, 1.0], [0.5, 0.5], [0.0, 1.0]],
            ],
        ]);

        // Shape: [batch=2, heads=3, seq_kv=3, head_dim=2]
        let key = query.clone();

        // Shape: [batch=2, heads=3, seq_kv=3, val_dim=2]
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
        // Shape: [batch=2, heads=3, seq_q=3, seq_kv=3]
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
                [
                    [false, false, true],
                    [false, false, true],
                    [false, false, true],
                ],
            ],
        ]);

        let output = attention(query, key, value, Some(mask));

        // Check output shape: [batch, heads, seq_q, val_dim]
        let shape = output.shape().dims();
        assert_eq!(shape, [2, 3, 3, 2]);
    }
}
