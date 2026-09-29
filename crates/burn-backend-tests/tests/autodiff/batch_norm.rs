use super::*;
use burn_tensor::Tolerance;
use burn_tensor::module::{batch_norm, batch_norm_train};

/// The closed-form gradient of `batch_norm_train` must match differentiating
/// through the batch statistics computed out of tensor operations, for the
/// input, gamma and beta alike.
#[test]
fn test_batch_norm_train_gradients_match_the_decomposed_statistics() {
    let device = AutodiffDevice::new();
    let tolerance = Tolerance::relative(1e-4).set_half_precision_relative(1e-2);

    let input_data = [
        [
            [0.9601, 0.7277, 0.1],
            [0.6272, 0.9034, 0.4],
            [0.9378, 0.7230, 0.2],
        ],
        [
            [0.6356, 0.1362, 0.7],
            [0.0249, 0.9509, 0.8],
            [0.6600, 0.5945, 0.3],
        ],
    ];
    let gamma_data = [2.0, 3.0, 4.0];
    let beta_data = [0.5, -1.0, 2.0];
    // A gradient that is not uniform, so every term of the closed form matters.
    let weights = TestTensor::<3>::from_data(
        [
            [[1.0, -2.0, 0.5], [0.3, 0.7, -1.1], [2.0, 0.1, 0.4]],
            [[-0.6, 1.4, 0.9], [1.2, -0.8, 0.2], [0.5, 0.5, -1.5]],
        ],
        &device,
    );

    let x = TestTensor::<3>::from_data(input_data, &device).require_grad();
    let gamma = TestTensor::<1>::from_data(gamma_data, &device).require_grad();
    let beta = TestTensor::<1>::from_data(beta_data, &device).require_grad();
    let result = batch_norm_train(x.clone(), gamma.clone(), beta.clone(), 1e-5);
    assert!(result.output.is_tracked());
    assert_eq!(result.mean.device(), device);
    assert_eq!(result.variance.device(), device);
    assert!(!result.mean.is_tracked());
    assert!(!result.variance.is_tracked());
    let grads = result.output.mul(weights.clone()).sum().backward();

    let x_ref = TestTensor::<3>::from_data(input_data, &device).require_grad();
    let gamma_ref = TestTensor::<1>::from_data(gamma_data, &device).require_grad();
    let beta_ref = TestTensor::<1>::from_data(beta_data, &device).require_grad();
    let mean = x_ref
        .clone()
        .swap_dims(0, 1)
        .reshape([3, 6])
        .mean_dim(1)
        .reshape([1, 3, 1]);
    let variance = x_ref
        .clone()
        .sub(mean.clone())
        .square()
        .swap_dims(0, 1)
        .reshape([3, 6])
        .mean_dim(1)
        .reshape([1, 3, 1]);
    let output_ref = batch_norm(
        x_ref.clone(),
        gamma_ref.clone(),
        beta_ref.clone(),
        mean.reshape([3]),
        variance.reshape([3]),
        1e-5,
    );
    let grads_ref = output_ref.mul(weights).sum().backward();

    x.grad(&grads)
        .unwrap()
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_ref.grad(&grads_ref).unwrap().to_data(), tolerance);
    gamma
        .grad(&grads)
        .unwrap()
        .to_data()
        .assert_approx_eq::<FloatElem>(&gamma_ref.grad(&grads_ref).unwrap().to_data(), tolerance);
    beta.grad(&grads)
        .unwrap()
        .to_data()
        .assert_approx_eq::<FloatElem>(&beta_ref.grad(&grads_ref).unwrap().to_data(), tolerance);
}

#[cfg(any(
    feature = "flex",
    feature = "vulkan",
    feature = "cuda",
    feature = "rocm",
    feature = "metal"
))]
#[test]
fn test_batch_norm_train_f16_input_gradient_with_large_reductions() {
    use burn_tensor::{DType, TensorData};

    let device = AutodiffDevice::new();
    // At 32768 samples the gradient sum overflows f16; at 131072 samples
    // the sample count overflows too. The input gradient remains representable.
    for batch in [2, 8] {
        let shape = [batch, 1, 128, 128];
        let values: Vec<f32> = (0..batch * 128 * 128)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let expected = TensorData::new(values.iter().map(|x| x * 0.75).collect::<Vec<_>>(), shape);
        let x = TestTensor::<4>::from_data(TensorData::new(values, shape), &device)
            .cast(DType::F16)
            .require_grad();
        let gamma = TestTensor::<1>::full([1], 2.0, &device).cast(DType::F16);
        let beta = TestTensor::<1>::zeros([1], &device).cast(DType::F16);
        let weights = x.clone().detach().add_scalar(2.0);

        // mean=0, variance=1, epsilon=3: normalized=x/2 and output=x.
        // With dy=x+2, dx=dy-mean(dy)-(x/2)*mean(dy*x/2)=3*x/4.
        let result = batch_norm_train(x.clone(), gamma, beta, 3.0);
        let grads = result.output.mul(weights).sum().backward();
        x.grad(&grads)
            .unwrap()
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-3));
    }
}
