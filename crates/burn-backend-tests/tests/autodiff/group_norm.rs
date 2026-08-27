use super::*;
use burn_tensor::{TensorData, Tolerance, module::group_norm};

fn input_data() -> TensorData {
    TensorData::from([
        [
            [1.0, -2.0, 3.0],
            [4.0, 0.5, -1.0],
            [2.0, 5.0, -3.0],
            [1.5, -4.0, 6.0],
        ],
        [
            [0.25, 3.5, -2.5],
            [-1.5, 2.25, 4.75],
            [7.0, -5.0, 1.0],
            [0.0, 2.5, -3.5],
        ],
    ])
}

#[test]
fn should_diff_group_norm_affine_combinations() {
    let device = AutodiffDevice::new();
    let expected_input_grad = TensorData::from([
        [
            [0.48075284, 0.39859906, 0.53552204],
            [-0.39403495, -0.48988104, -0.53095794],
            [0.36829582, 0.36159956, 0.3794563],
            [-0.37052792, -0.3582514, -0.38057235],
        ],
        [
            [0.34838018, 0.47799614, 0.2387051],
            [-0.4879666, -0.3384097, -0.2387051],
            [0.24915043, 0.42989415, 0.3395223],
            [-0.34454295, -0.3821979, -0.29182604],
        ],
    ]);
    let expected_gamma_grad = TensorData::from([-1.1733162, 1.1733162, 0.5757234, -0.5757234]);
    let zero_input_grad = TensorData::zeros::<f32, _>([2, 4, 3]);
    let expected_beta_grad = TensorData::from([6.0, 6.0, 6.0, 6.0]);
    let tolerance = Tolerance::rel_abs(1e-4, 1e-5)
        .set_half_precision_relative(1e-2)
        .set_half_precision_absolute(5e-3);

    for (scale, shift) in [(false, false), (true, false), (false, true), (true, true)] {
        let input = TestTensor::<3>::from_data(input_data(), &device).require_grad();
        let gamma = scale
            .then(|| TestTensor::<1>::from_data([0.5, -1.5, 2.0, -0.75], &device).require_grad());
        let beta = shift
            .then(|| TestTensor::<1>::from_data([0.25, -0.5, 1.0, 2.0], &device).require_grad());

        let output = group_norm(input.clone(), gamma.clone(), beta.clone(), 2, 1e-5);
        let grads = output.sum().backward();

        input
            .grad(&grads)
            .expect("input gradient should be present")
            .to_data()
            .assert_approx_eq::<FloatElem>(
                if scale {
                    &expected_input_grad
                } else {
                    &zero_input_grad
                },
                tolerance,
            );
        if let Some(gamma) = gamma {
            gamma
                .grad(&grads)
                .expect("gamma gradient should be present")
                .to_data()
                .assert_approx_eq::<FloatElem>(&expected_gamma_grad, tolerance);
        }
        if let Some(beta) = beta {
            beta.grad(&grads)
                .expect("beta gradient should be present")
                .to_data()
                .assert_approx_eq::<FloatElem>(&expected_beta_grad, tolerance);
        }
    }
}

#[test]
fn should_diff_group_norm_non_contiguous_input() {
    let device = AutodiffDevice::new();
    let storage_data = TensorData::from([
        [
            [1.0, 4.0, 2.0, 1.5],
            [-2.0, 0.5, 5.0, -4.0],
            [3.0, -1.0, -3.0, 6.0],
        ],
        [
            [0.25, -1.5, 7.0, 0.0],
            [3.5, 2.25, -5.0, 2.5],
            [-2.5, 4.75, 1.0, -3.5],
        ],
    ]);
    let upstream = TestTensor::<3>::from_data(
        TensorData::new(
            (1..=24).map(|value| value as f32 / 10.0).collect(),
            [2, 4, 3],
        ),
        &device,
    );

    let storage = TestTensor::<3>::from_data(storage_data, &device).require_grad();
    let gamma = TestTensor::<1>::from_data([0.5, -1.5, 2.0, -0.75], &device).require_grad();
    let beta = TestTensor::<1>::from_data([0.25, -0.5, 1.0, 2.0], &device).require_grad();
    let output = group_norm(
        storage.clone().swap_dims(1, 2),
        Some(gamma.clone()),
        Some(beta.clone()),
        2,
        1e-5,
    );
    let grads = (output * upstream).sum().backward();
    let input_grad = storage.grad(&grads).unwrap().swap_dims(1, 2).into_data();
    let gamma_grad = gamma.grad(&grads).unwrap().into_data();
    let beta_grad = beta.grad(&grads).unwrap().into_data();

    let input_ref = TestTensor::<3>::from_data(input_data(), &device).require_grad();
    let gamma_ref = TestTensor::<1>::from_data([0.5, -1.5, 2.0, -0.75], &device).require_grad();
    let beta_ref = TestTensor::<1>::from_data([0.25, -0.5, 1.0, 2.0], &device).require_grad();
    let output_ref = group_norm(
        input_ref.clone(),
        Some(gamma_ref.clone()),
        Some(beta_ref.clone()),
        2,
        1e-5,
    );
    let upstream_ref = TestTensor::<3>::from_data(
        TensorData::new(
            (1..=24).map(|value| value as f32 / 10.0).collect(),
            [2, 4, 3],
        ),
        &device,
    );
    let grads_ref = (output_ref * upstream_ref).sum().backward();

    let tolerance = Tolerance::rel_abs(1e-4, 1e-5)
        .set_half_precision_relative(1e-2)
        .set_half_precision_absolute(5e-3);
    input_grad
        .assert_approx_eq::<FloatElem>(&input_ref.grad(&grads_ref).unwrap().to_data(), tolerance);
    gamma_grad
        .assert_approx_eq::<FloatElem>(&gamma_ref.grad(&grads_ref).unwrap().to_data(), tolerance);
    beta_grad
        .assert_approx_eq::<FloatElem>(&beta_ref.grad(&grads_ref).unwrap().to_data(), tolerance);
}

#[test]
fn should_diff_group_norm_empty_input() {
    let device = AutodiffDevice::new();
    let input = TestTensor::<3>::empty([0, 4, 3], &device).require_grad();
    let gamma = TestTensor::<1>::ones([4], &device).require_grad();
    let beta = TestTensor::<1>::zeros([4], &device).require_grad();

    let output = group_norm(
        input.clone(),
        Some(gamma.clone()),
        Some(beta.clone()),
        2,
        1e-5,
    );
    let grads = output.sum().backward();

    let input_grad = input
        .grad(&grads)
        .expect("input gradient should be present");
    assert_eq!(input_grad.dims(), [0, 4, 3]);
    gamma
        .grad(&grads)
        .expect("gamma gradient should be present")
        .to_data()
        .assert_eq(&TensorData::zeros::<f32, _>([4]), false);
    beta.grad(&grads)
        .expect("beta gradient should be present")
        .to_data()
        .assert_eq(&TensorData::zeros::<f32, _>([4]), false);
}
