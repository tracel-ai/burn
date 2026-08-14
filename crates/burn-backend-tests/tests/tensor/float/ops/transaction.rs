use super::*;
use burn_tensor::Transaction;

// https://github.com/tracel-ai/burn/issues/4021
#[test]
fn should_support_transaction() {
    // The original reproduction used 261120 rows, but the regression only requires reading the
    // transposed (non-contiguous) 408x408 matmul output through a transaction.
    let rows = 32;
    let cols = 408;

    let device = Default::default();

    let j = TestTensor::<2>::zeros([rows, cols], &device);
    let jt = j.clone().transpose();

    let g = jt.matmul(j);

    let g = g.transpose();
    let expected = g.to_data();

    assert_eq!(g.shape().dims(), [cols, cols]);

    // Fails
    let [data] = Transaction::default()
        .register(g)
        .execute()
        .try_into()
        .unwrap();

    // check byte equality
    assert_eq!(data, expected);
}
