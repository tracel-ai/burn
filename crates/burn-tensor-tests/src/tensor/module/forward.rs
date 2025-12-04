use crate::*;
use burn_tensor::{TensorData, module::embedding};

#[test]
fn test_embedding_forward() {
    let weights = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let indices = TensorData::from([[0, 1], [1, 1]]);
    let weights = TestTensor::<2>::from(weights);
    let indices = TestTensorInt::<2>::from(indices);

    let output = embedding(weights, indices);
    let expected = TensorData::from([
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
    ]);

    output.into_data().assert_eq(&expected, false);
}
