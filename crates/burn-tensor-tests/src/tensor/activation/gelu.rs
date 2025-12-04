use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_gelu() {
    let tensor = TestTensor::<2>::from([[
        0.5447, 0.9809, 0.4114, 0.1398, 0.8045, 0.4103, 0.2388, 0.5262, 0.6677, 0.6737,
    ]]);
    let output = activation::gelu(tensor);
    let expected = TensorData::from([[
        0.3851, 0.8207, 0.2714, 0.0777, 0.6351, 0.2704, 0.1419, 0.3687, 0.4993, 0.5051,
    ]]);

    // Low precision to allow approximation implementation using tanh
    output.into_data().assert_approx_eq::<FloatElem>(
        &expected,
        Tolerance::default().set_half_precision_absolute(2e-3),
    );
}
