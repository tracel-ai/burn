use super::*;
use burn_tensor::module::rfft;
use burn_tensor::{DType, TensorData, Tolerance};

#[test]
fn rfft_of_tensor_dim2() {
    let signal = TestTensor::<2>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
    let dim = 0;
    let output = rfft(signal, dim);
    //let expected = TensorData::from([0.0, 0.146447, 0.5, 0.853553, 1.0, 0.853553, 0.5, 0.146447]);

    // Metal has less precise trigonometric functions.
    //let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    //tensor
    //    .into_data()
    //    .assert_approx_eq::<FloatElem>(&expected, tolerance);
}
