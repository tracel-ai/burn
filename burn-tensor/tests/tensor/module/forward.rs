use super::super::TestBackend;
use burn_tensor::{backend::Backend, module, Data, Tensor};

#[test]
fn test_embedding_forward() {
    let weights = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let indexes = Data::from([[0, 1], [1, 1]]);
    let weights = Tensor::<TestBackend, 2>::from_data(weights);
    let indexes = Tensor::<<TestBackend as Backend>::IntegerBackend, 2>::from_data(indexes);

    let output = module::embedding(&weights, &indexes);
    let expected = Data::from([
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
    ]);
    assert_eq!(output.to_data(), expected);
}
