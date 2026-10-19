use burn_backend::{Backend, Scalar, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_func::{Differentiable, FuncTensor, grad, hvp};

#[derive(Clone)]
struct Quartic;

impl<B: Backend> Differentiable<B, FuncTensor<B, 1>> for Quartic {
    fn eval<B2: Backend>(&self, x: &FuncTensor<B2, 1>) -> FuncTensor<B2, 1> {
        let x2 = x.clone() * x.clone();
        (x2.clone() * x2).sum()
    }
}

#[test]
fn hvp_of_quartic() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32]), &device);
    let v = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);

    let (gradient, hvp) = hvp::<Flex, _, _>(Quartic, &x, &v);

    // f(x) = x^4; f'(x) = 4x^3 = 32; f''(x) = 12x^2 = 48.
    assert_eq!(
        gradient.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![32.0]
    );
    assert_eq!(
        hvp.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![48.0]
    );
}

#[test]
fn hvp_matches_reverse_over_reverse() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.5f32]), &device);
    let v = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32]), &device);

    let (_, forward_over_reverse) = hvp::<Flex, _, _>(Quartic, &x, &v);

    // Reverse-over-reverse: differentiate x -> <grad(f)(x), v>.
    let v_for_inner = FuncTensor::<burn_autodiff::Autodiff<Flex>, 1>::from_data(
        TensorData::from([2.0f32]),
        &device,
    );
    let reverse_over_reverse =
        grad::<Flex, _, _>(move |x: &FuncTensor<burn_autodiff::Autodiff<Flex>, 1>| {
            let x2 = x.clone() * x.clone();
            let gradient = x2 * x.clone() * 4.0f64; // d(x^4)/dx = 4x^3
            (gradient * v_for_inner.clone()).sum()
        })(&x);

    let forward = forward_over_reverse
        .into_data_sync()
        .try_to_vec::<f32>()
        .unwrap()[0];
    let reverse = reverse_over_reverse
        .into_data_sync()
        .try_to_vec::<f32>()
        .unwrap()[0];
    assert!(
        (forward - reverse).abs() < 1e-4,
        "forward={forward}, reverse={reverse}"
    );
}

#[test]
fn second_derivative_of_transcendental() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.7f32]), &device);
    let v = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);

    #[derive(Clone)]
    struct Sin;
    impl<B: Backend> Differentiable<B, FuncTensor<B, 1>> for Sin {
        fn eval<B2: Backend>(&self, x: &FuncTensor<B2, 1>) -> FuncTensor<B2, 1> {
            x.clone().sin().sum()
        }
    }

    let (gradient, hvp) = hvp::<Flex, _, _>(Sin, &x, &v);
    let gradient = gradient.into_data_sync().try_to_vec::<f32>().unwrap()[0];
    let hvp = hvp.into_data_sync().try_to_vec::<f32>().unwrap()[0];
    assert!((gradient - 0.7f32.cos()).abs() < 1e-5);
    assert!((hvp + 0.7f32.sin()).abs() < 1e-5);
}

#[test]
fn scalar_gradient_of_mlp_loss() {
    use burn_autodiff::Autodiff;

    type AB = Autodiff<Flex>;
    let device = FlexDevice;
    let x = FuncTensor::<AB, 2>::from_data(TensorData::from([[0.2f32, -0.3], [0.5, 0.1]]), &device);
    let target = FuncTensor::<AB, 1>::from_data(TensorData::from([1.0f32, -1.0]), &device);

    let params = (
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[0.5f32, -0.2], [0.1, 0.3]]), &device),
        FuncTensor::<Flex, 1>::from_data(TensorData::from([0.0f32, 0.0]), &device),
    );

    let loss = |(w, b): &(FuncTensor<AB, 2>, FuncTensor<AB, 1>)| {
        let hidden = x.clone().matmul(w.clone()) + b.clone().reshape([1, 2]);
        let activated = hidden.tanh();
        let output = activated.sum_dim(1).reshape([2]);
        ((output - target.clone()).powf_scalar(Scalar::from(2.0f64))).mean()
    };

    let grads = grad::<Flex, _, _>(loss)(&params);
    assert_eq!(grads.0.shape().num_elements(), 4);
    assert_eq!(grads.1.shape().num_elements(), 2);
}

#[test]
fn second_derivative_of_linear_function_is_zero() {
    use burn_autodiff::Autodiff;

    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32, 4.0]), &device);

    // grad(grad(f)) where f(x) = sum(3x)
    let f = |x: &FuncTensor<Autodiff<Autodiff<Flex>>, 1>| (x.clone() * 3.0f64).sum();
    let d2 = grad::<Flex, _, _>(|x: &FuncTensor<Autodiff<Flex>, 1>| {
        grad::<Autodiff<Flex>, _, _>(f)(x).sum()
    })(&x);

    let values = d2.into_data_sync().try_to_vec::<f32>().unwrap();
    assert_eq!(values, vec![0.0, 0.0]);
}

#[test]
fn third_order_derivative_of_quartic() {
    use burn_autodiff::Autodiff;
    let device = FlexDevice;

    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.5f32]), &device);
    // f = sum(x^4) -> f' = 4x^3 -> f'' = 12x^2 -> f''' = 24x = 24 * 1.5 = 36.
    let f = |x: &FuncTensor<Autodiff<Autodiff<Autodiff<Flex>>>, 1>| {
        let x2 = x.clone() * x.clone();
        (x2.clone() * x2).sum()
    };
    let d3 = grad::<Flex, _, _>(|x: &FuncTensor<Autodiff<Flex>, 1>| {
        grad::<Autodiff<Flex>, _, _>(|x: &FuncTensor<Autodiff<Autodiff<Flex>>, 1>| {
            grad::<Autodiff<Autodiff<Flex>>, _, _>(f)(x).sum()
        })(x)
        .sum()
    })(&x);

    let val = d3.into_data_sync().try_to_vec::<f32>().unwrap()[0];
    assert!((val - 36.0).abs() < 1e-4, "Expected 36.0, got {val}");
}

#[test]
fn pinn_poisson_1d() {
    use burn_autodiff::Autodiff;
    use burn_backend::AutodiffBackend;

    type A1 = Autodiff<Flex>;
    type A2 = Autodiff<A1>;
    type A3 = Autodiff<A2>;

    let device = FlexDevice;
    let n = 16;
    let xs: Vec<f32> = (0..n)
        .map(|i| -1.0 + 2.0 * (i as f32) / ((n - 1) as f32))
        .collect();
    let x = FuncTensor::<Flex, 2>::from_data(TensorData::new(xs.clone(), [n, 1]), &device);

    // MLP: u(x) = W2 @ tanh(W1 @ x + b1) + b2
    let mut params = (
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[0.5f32], [0.7]]), &device),
        FuncTensor::<Flex, 1>::from_data(TensorData::from([0.0f32, 0.0]), &device),
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[0.5f32, 0.7]]), &device),
        FuncTensor::<Flex, 1>::from_data(TensorData::from([0.0f32]), &device),
    );

    fn mlp<B: Backend>(
        w1: &FuncTensor<B, 2>,
        b1: &FuncTensor<B, 1>,
        w2: &FuncTensor<B, 2>,
        b2: &FuncTensor<B, 1>,
        x: &FuncTensor<B, 2>,
    ) -> FuncTensor<B, 2> {
        let h = (x.clone().matmul(w1.clone().transpose()) + b1.clone().reshape([1, 2])).tanh();
        h.matmul(w2.clone().transpose()) + b2.clone().reshape([1, 1])
    }

    let compute_loss = |params: &(
        FuncTensor<A1, 2>,
        FuncTensor<A1, 1>,
        FuncTensor<A1, 2>,
        FuncTensor<A1, 1>,
    )| {
        let w1_2 =
            FuncTensor::<A2, 2>::from_primitive(A2::from_inner(params.0.primitive().clone()));
        let b1_2 =
            FuncTensor::<A2, 1>::from_primitive(A2::from_inner(params.1.primitive().clone()));
        let w2_2 =
            FuncTensor::<A2, 2>::from_primitive(A2::from_inner(params.2.primitive().clone()));
        let b2_2 =
            FuncTensor::<A2, 1>::from_primitive(A2::from_inner(params.3.primitive().clone()));

        let x_for_a1 = FuncTensor::<A1, 2>::from_primitive(A1::from_inner(x.primitive().clone()));

        let du_dx = move |x_inner: &FuncTensor<A2, 2>| -> FuncTensor<A2, 2> {
            let w1_3 =
                FuncTensor::<A3, 2>::from_primitive(A3::from_inner(w1_2.primitive().clone()));
            let b1_3 =
                FuncTensor::<A3, 1>::from_primitive(A3::from_inner(b1_2.primitive().clone()));
            let w2_3 =
                FuncTensor::<A3, 2>::from_primitive(A3::from_inner(w2_2.primitive().clone()));
            let b2_3 =
                FuncTensor::<A3, 1>::from_primitive(A3::from_inner(b2_2.primitive().clone()));
            grad::<A2, _, _>(move |x_in3: &FuncTensor<A3, 2>| {
                mlp(&w1_3, &b1_3, &w2_3, &b2_3, x_in3).sum()
            })(x_inner)
        };

        let du2_dx2 =
            grad::<A1, _, _>(move |x_in2: &FuncTensor<A2, 2>| du_dx(x_in2).sum())(&x_for_a1);

        let pi = core::f32::consts::PI;
        let pi_sq = pi * pi;
        let forcing = (x_for_a1.clone() * (pi as f64)).sin() * (pi_sq as f64);
        let residual = du2_dx2 + forcing;
        (residual.clone() * residual).mean()
    };

    // Initial loss
    let initial_loss = {
        let params_a1 = (
            FuncTensor::<A1, 2>::from_primitive(A1::from_inner(params.0.primitive().clone())),
            FuncTensor::<A1, 1>::from_primitive(A1::from_inner(params.1.primitive().clone())),
            FuncTensor::<A1, 2>::from_primitive(A1::from_inner(params.2.primitive().clone())),
            FuncTensor::<A1, 1>::from_primitive(A1::from_inner(params.3.primitive().clone())),
        );
        compute_loss(&params_a1)
            .into_data_sync()
            .try_to_vec::<f32>()
            .unwrap()[0]
    };

    // Compute gradients w.r.t. parameters
    let grads = grad::<Flex, _, _>(compute_loss)(&params);

    // Verify non-zero and finite gradients
    let g_w1 = grads
        .0
        .clone()
        .into_data_sync()
        .try_to_vec::<f32>()
        .unwrap();
    let g_b1 = grads
        .1
        .clone()
        .into_data_sync()
        .try_to_vec::<f32>()
        .unwrap();
    let g_w2 = grads
        .2
        .clone()
        .into_data_sync()
        .try_to_vec::<f32>()
        .unwrap();
    let g_b2 = grads
        .3
        .clone()
        .into_data_sync()
        .try_to_vec::<f32>()
        .unwrap();

    assert!(g_w1.iter().any(|&v| v != 0.0 && !v.is_nan()));
    assert!(g_b1.iter().all(|&v| !v.is_nan()));
    assert!(g_w2.iter().any(|&v| v != 0.0 && !v.is_nan()));
    assert!(g_b2.iter().all(|&v| !v.is_nan()));

    // Gradient descent step
    let lr = 1e-4f64;
    params.0 = params.0.clone() - grads.0 * lr;
    params.1 = params.1.clone() - grads.1 * lr;
    params.2 = params.2.clone() - grads.2 * lr;
    params.3 = params.3.clone() - grads.3 * lr;

    let updated_loss = {
        let params_a1 = (
            FuncTensor::<A1, 2>::from_primitive(A1::from_inner(params.0.primitive().clone())),
            FuncTensor::<A1, 1>::from_primitive(A1::from_inner(params.1.primitive().clone())),
            FuncTensor::<A1, 2>::from_primitive(A1::from_inner(params.2.primitive().clone())),
            FuncTensor::<A1, 1>::from_primitive(A1::from_inner(params.3.primitive().clone())),
        );
        compute_loss(&params_a1)
            .into_data_sync()
            .try_to_vec::<f32>()
            .unwrap()[0]
    };

    assert!(
        updated_loss < initial_loss,
        "Loss should decrease: initial={initial_loss}, updated={updated_loss}"
    );

    // Verify u is not all zeros
    let u_out = mlp(&params.0, &params.1, &params.2, &params.3, &x);
    let u_vals = u_out.into_data_sync().try_to_vec::<f32>().unwrap();
    assert!(u_vals.iter().any(|&v| v.abs() > 1e-3));
}

#[test]
fn hvp_fn_closure_of_quartic() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32]), &device);
    let v = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);

    let (gradient, hvp) = burn_func::hvp_fn::<Flex, _, _>(
        |x| {
            let x2 = x.clone() * x.clone();
            (x2.clone() * x2).sum()
        },
        &x,
        &v,
    );

    // f(x) = x^4; f'(x) = 4x^3 = 32; f''(x) = 12x^2 = 48.
    assert_eq!(
        gradient.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![32.0]
    );
    assert_eq!(
        hvp.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![48.0]
    );
}
