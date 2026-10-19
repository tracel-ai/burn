use burn_autodiff::Autodiff;
use burn_backend::{Backend, Scalar, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_func::{FuncTensor, grad, value_and_grad, vjp, vjp_once};

type AB = Autodiff<Flex>;

fn tensor(values: &[f32]) -> FuncTensor<Flex, 1> {
    FuncTensor::from_data(TensorData::from(values), &FlexDevice)
}

fn values(tensor: FuncTensor<Flex, 1>) -> Vec<f32> {
    tensor.into_data_sync().try_to_vec::<f32>().unwrap()
}

#[test]
fn issue_6090_stateless_line_fit() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 2.0, 3.0, 4.0]), &device);
    let y = FuncTensor::<Flex, 1>::from_data(TensorData::from([5.0f32, 7.0, 9.0, 11.0]), &device);
    let x_ad = FuncTensor::<AB, 1>::from_data(TensorData::from([1.0f32, 2.0, 3.0, 4.0]), &device);
    let y_ad = FuncTensor::<AB, 1>::from_data(TensorData::from([5.0f32, 7.0, 9.0, 11.0]), &device);

    let mut params = (tensor(&[0.0]), tensor(&[0.0]));
    let loss_fn = move |(w, b): &(FuncTensor<AB, 1>, FuncTensor<AB, 1>)| {
        let predictions = w.clone() * x_ad.clone() + b.clone();
        (predictions - y_ad.clone())
            .powf_scalar(Scalar::from(2.0f64))
            .mean()
    };

    let value_and_gradient = value_and_grad::<Flex, AB, _, _>(loss_fn);
    for _ in 0..1000 {
        let (_, grads) = value_and_gradient(&params);
        params.0 = params.0 - grads.0 * 0.1f64;
        params.1 = params.1 - grads.1 * 0.1f64;
    }

    let w = values(params.0)[0];
    let b = values(params.1)[0];
    assert!((w - 2.0).abs() < 1e-2, "expected w=2, got {w}");
    assert!((b - 3.0).abs() < 1e-2, "expected b=3, got {b}");

    let _ = (x, y);
}

#[test]
fn constant_function_returns_zero_gradient() {
    let x = tensor(&[2.0, 4.0]);
    let f = |_x: &FuncTensor<AB, 1>| FuncTensor::<AB, 1>::ones([1], &FlexDevice);
    let dx = grad::<Flex, _, _>(f)(&x);
    assert_eq!(values(dx), vec![0.0, 0.0]);
}

#[test]
fn unused_pytree_leaf_gets_zero_gradient() {
    let params = (tensor(&[2.0]), tensor(&[5.0, 6.0]));
    let f = |(w, _b): &(FuncTensor<AB, 1>, FuncTensor<AB, 1>)| (w.clone() * 3.0f64).sum();
    let grads = grad::<Flex, _, _>(f)(&params);
    assert_eq!(values(grads.0), vec![3.0]);
    assert_eq!(values(grads.1), vec![0.0, 0.0]);
}

#[test]
fn aliased_pytree_leaves_each_receive_gradient() {
    let x = tensor(&[2.0]);
    let params = (x.clone(), x);
    let f = |(a, b): &(FuncTensor<AB, 1>, FuncTensor<AB, 1>)| {
        (a.clone() * 2.0f64 + b.clone() * 3.0f64).sum()
    };
    let grads = grad::<Flex, _, _>(f)(&params);
    assert_eq!(values(grads.0), vec![2.0]);
    assert_eq!(values(grads.1), vec![3.0]);
}

#[test]
fn multi_output_vjp_sums_cotangents() {
    let x = tensor(&[2.0]);
    let f = |x: &FuncTensor<AB, 1>| (x.clone() * 2.0f64, x.clone() * 3.0f64);

    let (output, pullback) =
        vjp::<Flex, AB, _, (FuncTensor<Flex, 1>, FuncTensor<Flex, 1>), _>(f, &x);
    assert_eq!(values(output.0), vec![4.0]);
    assert_eq!(values(output.1), vec![6.0]);

    let cotangent = (tensor(&[1.0]), tensor(&[1.0]));
    let dx = pullback(&cotangent);
    assert_eq!(values(dx), vec![5.0]);
}

#[test]
fn vjp_shared_intermediate_nodes() {
    let x = tensor(&[2.0]);
    let f = |x: &FuncTensor<AB, 1>| {
        let shared = x.clone() * x.clone();
        (shared.clone(), shared)
    };

    let (_, pullback) = vjp::<Flex, AB, _, (FuncTensor<Flex, 1>, FuncTensor<Flex, 1>), _>(f, &x);

    let cotangent = (tensor(&[1.0]), tensor(&[2.0]));
    let dx = pullback(&cotangent);
    // d(x^2)/dx = 2x = 4; two outputs contribute 4 * (1 + 2) = 12.
    assert_eq!(values(dx), vec![12.0]);
}

#[test]
fn vjp_once_runs_once() {
    let x = tensor(&[3.0]);
    let f = |x: &FuncTensor<AB, 1>| x.clone() * x.clone();
    let (output, pullback) = vjp_once::<Flex, AB, _, FuncTensor<Flex, 1>, _>(f, &x);
    assert_eq!(values(output), vec![9.0]);
    let dx = pullback(&tensor(&[1.0]));
    assert_eq!(values(dx), vec![6.0]);
}

#[test]
fn finite_difference_sweep() {
    fn assert_fd<Fa, Fb>(name: &str, x0: &[f32], f_ad: Fa, f_fd: Fb, tol: f64)
    where
        Fa: Fn(&FuncTensor<AB, 1>) -> FuncTensor<AB, 1>,
        Fb: Fn(&FuncTensor<Flex, 1>) -> FuncTensor<Flex, 1>,
    {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from(x0), &device);
        let analytical = values(grad::<Flex, _, _>(&f_ad)(&x));

        let eps = 1e-2f64;
        let mut finite = Vec::new();
        for index in 0..x0.len() {
            let mut plus = x0.to_vec();
            let mut minus = x0.to_vec();
            plus[index] += eps as f32;
            minus[index] -= eps as f32;
            let plus_value = values(f_fd(&FuncTensor::from_data(
                TensorData::from(plus.as_slice()),
                &device,
            )))[0];
            let minus_value = values(f_fd(&FuncTensor::from_data(
                TensorData::from(minus.as_slice()),
                &device,
            )))[0];
            finite.push((plus_value - minus_value) as f64 / (2.0 * eps));
        }

        for (a, f) in analytical.iter().zip(finite.iter()) {
            assert!(
                ((*a as f64) - f).abs() < tol,
                "{name}: analytic {a} vs fd {f} with tol {tol}"
            );
        }
    }

    // Arithmetic & unary
    assert_fd(
        "exp+sin+x2",
        &[0.3, -0.7, 1.1],
        |x| (x.clone().exp() + x.clone().sin() + x.clone() * x.clone()).sum(),
        |x| (x.clone().exp() + x.clone().sin() + x.clone() * x.clone()).sum(),
        1e-3,
    );

    assert_fd(
        "relu",
        &[-1.0, 0.5, 2.0],
        |x| x.clone().relu().sum(),
        |x| x.clone().relu().sum(),
        1e-3,
    );

    assert_fd(
        "gelu",
        &[-1.0, 0.5, 2.0],
        |x| x.clone().gelu().sum(),
        |x| x.clone().gelu().sum(),
        1e-2,
    );

    assert_fd(
        "sigmoid",
        &[-1.0, 0.0, 1.0],
        |x| x.clone().sigmoid().sum(),
        |x| x.clone().sigmoid().sum(),
        1e-3,
    );

    assert_fd(
        "tanh",
        &[-0.5, 0.0, 0.5],
        |x| x.clone().tanh().sum(),
        |x| x.clone().tanh().sum(),
        1e-3,
    );

    assert_fd(
        "softmax",
        &[-0.3, 0.0, 0.7],
        |x| x.clone().softmax(0).sum(),
        |x| x.clone().softmax(0).sum(),
        1e-3,
    );

    assert_fd(
        "log_softmax",
        &[-0.3, 0.0, 0.7],
        |x| x.clone().log_softmax(0).sum(),
        |x| x.clone().log_softmax(0).sum(),
        1e-3,
    );

    // Reductions
    assert_fd(
        "sum_dim",
        &[1.0, 2.0, 3.0],
        |x| x.clone().sum_dim(0).sum(),
        |x| x.clone().sum_dim(0).sum(),
        1e-3,
    );

    // Structural
    assert_fd(
        "reshape",
        &[1.0, 2.0, 3.0, 4.0],
        |x| x.clone().reshape::<2, _>([2, 2]).sum(),
        |x| x.clone().reshape::<2, _>([2, 2]).sum(),
        1e-3,
    );

    assert_fd(
        "slice",
        &[1.0, 2.0, 3.0, 4.0],
        |x| {
            x.clone()
                .slice(&[burn_std::Slice::new(0, Some(2), 1)])
                .sum()
        },
        |x| {
            x.clone()
                .slice(&[burn_std::Slice::new(0, Some(2), 1)])
                .sum()
        },
        1e-3,
    );

    assert_fd(
        "cat",
        &[1.0, 2.0, 3.0],
        |x| FuncTensor::cat(vec![x.clone(), x.clone()], 0).sum(),
        |x| FuncTensor::cat(vec![x.clone(), x.clone()], 0).sum(),
        1e-3,
    );

    // Broadcasting
    assert_fd(
        "broadcasting",
        &[1.0, 2.0, 3.0],
        |x| {
            let b = FuncTensor::<AB, 2>::from_data(
                TensorData::from([[1.0f32, 1.0, 1.0], [2.0, 2.0, 2.0]]),
                &FlexDevice,
            );
            (x.clone().reshape::<2, _>([1, 3]) + b).sum()
        },
        |x| {
            let b = FuncTensor::<Flex, 2>::from_data(
                TensorData::from([[1.0f32, 1.0, 1.0], [2.0, 2.0, 2.0]]),
                &FlexDevice,
            );
            (x.clone().reshape::<2, _>([1, 3]) + b).sum()
        },
        1e-3,
    );

    // Matmul
    assert_fd(
        "matmul",
        &[1.0, 2.0, 3.0, 4.0],
        |x| {
            let w = FuncTensor::<AB, 2>::from_data(
                TensorData::from([[0.5f32, -0.2], [0.1, 0.3]]),
                &FlexDevice,
            );
            x.clone().reshape::<2, _>([2, 2]).matmul(w).sum()
        },
        |x| {
            let w = FuncTensor::<Flex, 2>::from_data(
                TensorData::from([[0.5f32, -0.2], [0.1, 0.3]]),
                &FlexDevice,
            );
            x.clone().reshape::<2, _>([2, 2]).matmul(w).sum()
        },
        1e-3,
    );
}

#[test]
fn grad_rejects_non_scalar_output() {
    let x = tensor(&[1.0, 2.0]);
    let f = |x: &FuncTensor<AB, 1>| x.clone() * 2.0f64;
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| grad::<Flex, _, _>(f)(&x)));
    assert!(result.is_err());
}

#[test]
fn backend_is_autodiff_capable() {
    assert!(!<Flex as Backend>::ad_enabled(&FlexDevice));
    assert!(<AB as Backend>::ad_enabled(&FlexDevice));
}

#[test]
#[should_panic(expected = "Cotangent shape")]
fn vjp_mismatched_cotangent_shape_panics() {
    let device = FlexDevice;
    let x =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]), &device);
    let wrong_seed = FuncTensor::<Flex, 2>::from_data(TensorData::from([[1.0f32, 1.0]]), &device);
    let (_, pullback) = vjp::<Flex, AB, _, _, _>(|x| x.clone(), &x);
    let _ = pullback(&wrong_seed);
}

#[test]
fn vjp_untracked_output_returns_zero() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32, 3.0]), &device);
    // Function that returns a constant untracked tensor (ignoring input)
    let (_, pullback) = vjp::<Flex, AB, _, _, _>(
        |x| FuncTensor::from_data(TensorData::from([5.0f32]), &x.device()),
        &x,
    );
    let seed = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);
    let grad = pullback(&seed);
    assert_eq!(values(grad), vec![0.0, 0.0]);
}
