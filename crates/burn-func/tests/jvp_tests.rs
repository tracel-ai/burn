use burn_autodiff::Autodiff;
use burn_backend::{Scalar, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_func::{ForwardDiff, FuncTensor, jvp, value_and_jvp, vjp};

type AB = Autodiff<Flex>;
type FD = ForwardDiff<Flex>;

fn values(tensor: FuncTensor<Flex, 1>) -> Vec<f32> {
    tensor.into_data_sync().try_to_vec::<f32>().unwrap()
}

#[test]
fn jvp_arithmetic_and_reduction() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.5f32, -0.25]), &device);
    let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 2.0]), &device);

    let f = |x: &FuncTensor<FD, 1>| (x.clone() * x.clone() + x.clone().sin()).sum();
    let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);

    // d/dx (x^2 + sin x) = 2x + cos x
    let expected = (2.0 * 0.5 + 0.5f32.cos()) * 1.0 + (2.0 * -0.25 + (-0.25f32).cos()) * 2.0;
    assert!((values(tangent)[0] - expected).abs() < 1e-5);
}

#[test]
fn jvp_broadcasted_tangent() {
    let device = FlexDevice;
    let x =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]), &device);
    let dx =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[1.0f32, 0.0], [0.0, 1.0]]), &device);
    let b = FuncTensor::<Flex, 2>::from_data(TensorData::from([[2.0f32, 3.0]]), &device);
    let db = FuncTensor::<Flex, 2>::from_data(TensorData::from([[0.5f32, -0.5]]), &device);

    let f = |(x, b): &(FuncTensor<FD, 2>, FuncTensor<FD, 2>)| (x.clone() * b.clone()).sum();
    let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &(x, b), &(dx, db));

    // sum(dx * b) + sum(x * db) = (2 + 3) + (0.5 - 2 + 1.5 - 2) = 5 - 1 = 4
    assert!((values(tangent)[0] - 4.0).abs() < 1e-5);
}

#[test]
fn jvp_powf_negative_base_constant_exponent() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([-2.0f32]), &device);
    let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);
    let exponent = FuncTensor::<FD, 1>::from_data(TensorData::from([3.0f32]), &device);

    let f = move |x: &FuncTensor<FD, 1>| x.clone().powf(exponent.clone()).sum();
    let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);

    // d/dx x^3 at x = -2 is 3 * 4 = 12. The constant exponent must not introduce NaN.
    let value = values(tangent)[0];
    assert!(value.is_finite());
    assert!((value - 12.0).abs() < 1e-5);
}

#[test]
fn adjoint_equivalence_for_matmul() {
    let device = FlexDevice;
    let x =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]), &device);
    let w =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[0.5f32, -0.5], [1.0, 2.0]]), &device);
    let dx =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[0.1f32, 0.2], [0.3, 0.4]]), &device);
    let dw =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[0.5f32, 0.6], [0.7, 0.8]]), &device);
    let cotangent =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]), &device);

    let f = |(x, w): &(FuncTensor<AB, 2>, FuncTensor<AB, 2>)| x.clone().matmul(w.clone());

    // jvp(f, (x,w), (dx,dw))
    let f_jvp = |(x, w): &(FuncTensor<FD, 2>, FuncTensor<FD, 2>)| x.clone().matmul(w.clone());
    let jvp_out: FuncTensor<Flex, 2> =
        jvp::<Flex, _, _, _>(f_jvp, &(x.clone(), w.clone()), &(dx.clone(), dw.clone()));

    // vjp(f, (x,w)) with cotangent
    let primals = (x, w);
    let (_, pullback) = vjp::<Flex, AB, _, FuncTensor<Flex, 2>, _>(f, &primals);
    let (grad_x, grad_w) = pullback(&cotangent);

    let lhs = dot(jvp_out, cotangent);
    let rhs = dot(dx, grad_x) + dot(dw, grad_w);

    assert!((lhs - rhs).abs() < 1e-5, "lhs={lhs}, rhs={rhs}");
}

fn dot(lhs: FuncTensor<Flex, 2>, rhs: FuncTensor<Flex, 2>) -> f32 {
    let product = (lhs * rhs).sum();
    values(product)[0]
}

#[test]
fn value_and_jvp_returns_primal_and_tangent() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32]), &device);
    let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);
    let f = |x: &FuncTensor<FD, 1>| (x.clone() * 4.0f64).sum();

    let (value, tangent): (FuncTensor<Flex, 1>, FuncTensor<Flex, 1>) =
        value_and_jvp::<Flex, _, _, _>(f, &x, &u);
    assert_eq!(values(value), vec![8.0]);
    assert_eq!(values(tangent), vec![4.0]);
}

#[test]
fn jvp_scalar_power() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32]), &device);
    let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);
    let f = |x: &FuncTensor<FD, 1>| x.clone().powf_scalar(Scalar::from(3.0f64)).sum();
    let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);
    assert!((values(tangent)[0] - 12.0).abs() < 1e-5);
}

#[test]
#[allow(clippy::type_complexity)]
fn jvp_unary_functions() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.5f32]), &device);
    let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);

    let checks: Vec<(
        &str,
        Box<dyn Fn(&FuncTensor<FD, 1>) -> FuncTensor<FD, 1>>,
        f32,
    )> = vec![
        (
            "exp",
            Box::new(|x: &FuncTensor<FD, 1>| x.clone().exp().sum()),
            0.5f32.exp(),
        ),
        (
            "log",
            Box::new(|x: &FuncTensor<FD, 1>| x.clone().log().sum()),
            2.0,
        ),
        (
            "sqrt",
            Box::new(|x: &FuncTensor<FD, 1>| x.clone().sqrt().sum()),
            1.0 / (2.0 * 0.5f32.sqrt()),
        ),
        (
            "cos",
            Box::new(|x: &FuncTensor<FD, 1>| x.clone().cos().sum()),
            -0.5f32.sin(),
        ),
        (
            "tanh",
            Box::new(|x: &FuncTensor<FD, 1>| x.clone().tanh().sum()),
            1.0 - 0.5f32.tanh().powi(2),
        ),
        (
            "sin",
            Box::new(|x: &FuncTensor<FD, 1>| x.clone().sin().sum()),
            0.5f32.cos(),
        ),
        (
            "tan",
            Box::new(|x: &FuncTensor<FD, 1>| x.clone().tan().sum()),
            1.0 + 0.5f32.tan().powi(2),
        ),
        (
            "sinh",
            Box::new(|x: &FuncTensor<FD, 1>| x.clone().sinh().sum()),
            0.5f32.cosh(),
        ),
        (
            "cosh",
            Box::new(|x: &FuncTensor<FD, 1>| x.clone().cosh().sum()),
            0.5f32.sinh(),
        ),
        (
            "erf",
            Box::new(|x: &FuncTensor<FD, 1>| x.clone().erf().sum()),
            2.0 / core::f32::consts::PI.sqrt() * (-(0.5f32.powi(2))).exp(),
        ),
        (
            "abs",
            Box::new(|x: &FuncTensor<FD, 1>| x.clone().abs().sum()),
            1.0,
        ),
        (
            "recip",
            Box::new(|x: &FuncTensor<FD, 1>| x.clone().recip().sum()),
            -1.0 / (0.5f32.powi(2)),
        ),
    ];

    for (name, f, expected) in checks {
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(&f, &x, &u);
        let actual = values(tangent)[0];
        assert!(
            (actual - expected).abs() < 1e-4,
            "{name}: {actual} vs {expected}"
        );
    }
}

#[test]
fn jvp_cumprod_matches_finite_difference() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32, 3.0, 4.0]), &device);
    let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 0.0, 0.0]), &device);
    let f = |x: &FuncTensor<FD, 1>| x.clone().cumprod(0).sum();
    let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);
    assert!((values(tangent)[0] - 16.0).abs() < 1e-3);
}

#[test]
fn jvp_cummin_matches_finite_difference() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([3.0f32, 1.0, 2.0]), &device);
    let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.0f32, 1.0, 0.0]), &device);
    let f = |x: &FuncTensor<FD, 1>| x.clone().cummin(0).sum();
    let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);
    assert!((values(tangent)[0] - 2.0).abs() < 1e-5);
}

#[test]
fn jvp_cummax_matches_finite_difference() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 3.0, 2.0]), &device);
    let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.0f32, 1.0, 0.0]), &device);
    let f = |x: &FuncTensor<FD, 1>| x.clone().cummax(0).sum();
    let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);
    assert!((values(tangent)[0] - 2.0).abs() < 1e-5);
}

#[test]
fn jvp_attention_smoke() {
    let device = FlexDevice;
    let q = FuncTensor::<Flex, 4>::from_data(
        TensorData::from([[[[1.0f32, 0.0], [0.0, 1.0]]]]),
        &device,
    );
    let k = FuncTensor::<Flex, 4>::from_data(
        TensorData::from([[[[1.0f32, 0.0], [0.0, 1.0]]]]),
        &device,
    );
    let v = FuncTensor::<Flex, 4>::from_data(
        TensorData::from([[[[1.0f32, 2.0], [3.0, 4.0]]]]),
        &device,
    );

    let dq = FuncTensor::<Flex, 4>::from_data(
        TensorData::from([[[[0.1f32, 0.0], [0.0, 0.0]]]]),
        &device,
    );
    let dk = FuncTensor::<Flex, 4>::zeros([1, 1, 2, 2], &device);
    let dv = FuncTensor::<Flex, 4>::zeros([1, 1, 2, 2], &device);

    let f = |(q, k, v): &(FuncTensor<FD, 4>, FuncTensor<FD, 4>, FuncTensor<FD, 4>)| {
        q.clone().attention(k.clone(), v.clone()).sum()
    };
    let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &(q, k, v), &(dq, dk, dv));
    let val = values(tangent)[0];
    assert!(val.is_finite());
    assert!(val.abs() > 1e-4);
}

#[test]
fn jvp_grid_sample_2d_smoke() {
    use burn_backend::ops::{GridSampleOptions, InterpolateMode};
    let device = FlexDevice;
    let tensor = FuncTensor::<Flex, 4>::from_data(
        TensorData::from([[[[1.0f32, 2.0], [3.0, 4.0]]]]),
        &device,
    );
    let grid = FuncTensor::<Flex, 4>::from_data(
        TensorData::from([[[[-0.5f32, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]]),
        &device,
    );
    let dt = FuncTensor::<Flex, 4>::from_data(
        TensorData::from([[[[1.0f32, 1.0], [1.0, 1.0]]]]),
        &device,
    );
    let dg = FuncTensor::<Flex, 4>::zeros([1, 2, 2, 2], &device);

    let options = GridSampleOptions::new(InterpolateMode::Bilinear);
    let f = |(t, g): &(FuncTensor<FD, 4>, FuncTensor<FD, 4>)| {
        t.clone().grid_sample_2d(g.clone(), options.clone()).sum()
    };
    let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &(tensor, grid), &(dt, dg));
    let val = values(tangent)[0];
    assert!(val.is_finite());
    assert!((val - 4.0).abs() < 1e-3);
}

#[test]
fn jvp_clamp() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.0f32, 0.5, 2.0]), &device);
    let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 1.0, 1.0]), &device);
    let f = |x: &FuncTensor<FD, 1>| x.clone().clamp(0.2f64.into(), 1.0f64.into()).sum();
    let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);
    assert!((values(tangent)[0] - 1.0).abs() < 1e-5);
}

#[test]
fn jvp_max_dim_and_min_dim() {
    let device = FlexDevice;
    let x =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[1.0f32, 5.0], [4.0, 2.0]]), &device);
    let u =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[0.1f32, 0.2], [0.3, 0.4]]), &device);

    let f_max = |x: &FuncTensor<FD, 2>| x.clone().max_dim(1).sum();
    let t_max: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f_max, &x, &u);
    assert!((values(t_max)[0] - 0.5).abs() < 1e-5);

    let f_min = |x: &FuncTensor<FD, 2>| x.clone().min_dim(1).sum();
    let t_min: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f_min, &x, &u);
    assert!((values(t_min)[0] - 0.5).abs() < 1e-5);
}

#[test]
fn jvp_slice_assign() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 2.0, 3.0, 4.0]), &device);
    let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 1.0, 1.0, 1.0]), &device);
    let v = FuncTensor::<Flex, 1>::from_data(TensorData::from([10.0f32, 20.0]), &device);
    let dv = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32, 3.0]), &device);

    let f = |(x, v): &(FuncTensor<FD, 1>, FuncTensor<FD, 1>)| {
        x.clone()
            .slice_assign(&[burn_std::Slice::new(1, Some(3), 1)], v.clone())
            .sum()
    };
    let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &(x, v), &(u, dv));
    assert!((values(tangent)[0] - 7.0).abs() < 1e-5);
}

#[test]
fn jvp_mask_where_and_fill() {
    use burn_backend::ops::{BoolTensorOps, FloatTensorOps};
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 2.0, 3.0]), &device);
    let y = FuncTensor::<Flex, 1>::from_data(TensorData::from([10.0f32, 20.0, 30.0]), &device);
    let dx = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 2.0, 3.0]), &device);
    let dy = FuncTensor::<Flex, 1>::from_data(TensorData::from([10.0f32, 20.0, 30.0]), &device);
    let mask = burn_flex::Flex::bool_from_data(TensorData::from([true, false, true]), &device);

    let mask_clone = mask.clone();
    let f_where = move |(x, y): &(FuncTensor<FD, 1>, FuncTensor<FD, 1>)| {
        let prim = FD::float_mask_where(
            x.primitive().clone(),
            mask_clone.clone(),
            y.primitive().clone(),
        );
        FuncTensor::<FD, 1>::from_primitive(prim).sum()
    };
    let tangent_where: FuncTensor<Flex, 1> =
        jvp::<Flex, _, _, _>(f_where, &(x.clone(), y), &(dx.clone(), dy));
    assert!((values(tangent_where)[0] - 42.0).abs() < 1e-5);

    let f_fill = move |x: &FuncTensor<FD, 1>| {
        let prim = FD::float_mask_fill(x.primitive().clone(), mask.clone(), 0.0f64.into());
        FuncTensor::<FD, 1>::from_primitive(prim).sum()
    };
    let tangent_fill: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f_fill, &x, &dx);
    assert!((values(tangent_fill)[0] - 2.0).abs() < 1e-5);
}

#[test]
fn jvp_sort_and_topk() {
    use burn_backend::ops::FloatTensorOps;
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([3.0f32, 1.0, 4.0, 2.0]), &device);
    let dx = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.1f32, 0.2, 0.3, 0.4]), &device);

    let f_sort = |x: &FuncTensor<FD, 1>| {
        let prim = FD::float_sort(x.primitive().clone(), 0, false);
        FuncTensor::<FD, 1>::from_primitive(prim)
    };
    let t_sort: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f_sort, &x, &dx);
    assert_eq!(values(t_sort), vec![0.2, 0.4, 0.1, 0.3]);

    let f_topk = |x: &FuncTensor<FD, 1>| {
        let prim = FD::float_topk(x.primitive().clone(), 0, 2);
        FuncTensor::<FD, 1>::from_primitive(prim)
    };
    let t_topk: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f_topk, &x, &dx);
    assert_eq!(values(t_topk), vec![0.3, 0.1]);
}

#[test]
fn jvp_pad() {
    use burn_backend::ops::FloatTensorOps;
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32, 3.0]), &device);
    let dx = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.5f32, 1.5]), &device);

    let f_pad = |x: &FuncTensor<FD, 1>| {
        let prim = FD::float_pad(x.primitive().clone(), &[(1, 1)], 0.0f64.into());
        FuncTensor::<FD, 1>::from_primitive(prim).sum()
    };
    let t_pad: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f_pad, &x, &dx);
    assert!((values(t_pad)[0] - 2.0).abs() < 1e-5);
}

#[test]
fn jvp_powf_zero_base() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.0f32]), &device);
    let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);
    let exponent = FuncTensor::<FD, 1>::from_data(TensorData::from([3.0f32]), &device);

    let f = move |x: &FuncTensor<FD, 1>| x.clone().powf(exponent.clone()).sum();
    let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);

    // d/dx x^3 at x = 0 is 3 * 0^2 = 0.0. Must be finite, not NaN.
    let value = values(tangent)[0];
    assert!(value.is_finite());
    assert_eq!(value, 0.0);
}

#[test]
fn jvp_max_pool2d() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 4>::from_data(
        TensorData::from([[[[1.0f32, 4.0], [2.0, 3.0]]]]),
        &device,
    );
    let dx = FuncTensor::<Flex, 4>::from_data(
        TensorData::from([[[[10.0f32, 20.0], [30.0, 40.0]]]]),
        &device,
    );

    let f = |x: &FuncTensor<FD, 4>| x.clone().max_pool2d([2, 2], [2, 2], [0, 0], [1, 1]);
    let tangent: FuncTensor<Flex, 4> = jvp::<Flex, _, _, _>(f, &x, &dx);

    let vals = tangent.into_data_sync().try_to_vec::<f32>().unwrap();
    // Max element is 4.0 at (0, 1), so winning tangent is 20.0
    assert_eq!(vals, vec![20.0]);
}
