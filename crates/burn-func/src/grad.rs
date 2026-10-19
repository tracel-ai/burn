use burn_autodiff::Autodiff;
use burn_backend::{AutodiffBackend, Backend, Shape, TensorMetadata, tensor::FloatTensor};

use crate::{FuncTensor, PyTree, forward_diff::ForwardDiff, forward_diff::value_and_jvp};

/// Computes the gradient of a scalar-valued function with respect to its parameter tree.
///
/// The returned closure takes the parameters on `B`, lifts them to `Autodiff<B>`, runs `f`,
/// seeds the backward pass with ones, and rebuilds a parameter tree on `B`.
///
/// Leaves that `f` does not use receive zero gradients matching their shape, device, and dtype.
/// Aliased leaves (the same tensor appearing in multiple tree positions) each receive the full
/// gradient because the extraction is non-destructive.
///
/// # Panics
///
/// Panics if the function output is not scalar-valued (`num_elements() != 1`).
pub fn grad<B, F, P>(f: F) -> impl Fn(&P) -> P
where
    B: Backend,
    P: PyTree<B, WithBackend<B> = P>,
    F: Fn(&P::WithBackend<Autodiff<B>>) -> FuncTensor<Autodiff<B>, 1>,
{
    move |params| value_and_grad_impl::<B, Autodiff<B>, F, P>(&f, params).1
}

/// Computes both the value and the gradient of a scalar-valued function.
///
/// `AB` is normally `Autodiff<B>`. It is a parameter so callers can select a checkpointing
/// strategy or a different reverse-mode decorator.
///
/// # Panics
///
/// Panics if the function output is not scalar-valued (`num_elements() != 1`).
pub fn value_and_grad<B, AB, F, P>(f: F) -> impl Fn(&P) -> (FuncTensor<B, 1>, P)
where
    B: Backend,
    AB: AutodiffBackend<InnerBackend = B>,
    P: PyTree<B, WithBackend<B> = P>,
    F: Fn(&P::WithBackend<AB>) -> FuncTensor<AB, 1>,
{
    move |params| value_and_grad_impl::<B, AB, F, P>(&f, params)
}

fn value_and_grad_impl<B, AB, F, P>(f: &F, params: &P) -> (FuncTensor<B, 1>, P)
where
    B: Backend,
    AB: AutodiffBackend<InnerBackend = B>,
    P: PyTree<B, WithBackend<B> = P>,
    F: Fn(&P::WithBackend<AB>) -> FuncTensor<AB, 1>,
{
    let lifted = lift::<B, AB, P>(params);
    let loss_ad = f(&lifted);

    assert_eq!(
        loss_ad.num_elements(),
        1,
        "grad requires a scalar-valued output tensor (num_elements == 1), got shape {:?}",
        loss_ad.shape()
    );

    let loss_primitive = loss_ad.into_primitive();

    // A constant function (or one whose output was detached) leaves the loss untracked. There is
    // no tape to traverse; the gradient is zero for every parameter leaf.
    if !AB::is_tracked(&loss_primitive) {
        let loss_inner = AB::inner(loss_primitive);
        return (
            FuncTensor::from_primitive(loss_inner),
            zero_like::<B, P>(params),
        );
    }

    let loss_inner = AB::inner(loss_primitive.clone());
    let gradients = AB::backward(loss_primitive);

    let parameter_grads = params.zip_float::<AB, B, _>(
        &lifted,
        &mut |original: &FloatTensor<B>, lifted_leaf: &FloatTensor<AB>| {
            // `grad` is non-destructive: aliased leaves each read the same registered gradient.
            AB::grad(lifted_leaf, &gradients).unwrap_or_else(|| {
                B::float_zeros(
                    original.shape(),
                    &original.device(),
                    original.dtype().into(),
                )
            })
        },
    );

    (FuncTensor::from_primitive(loss_inner), parameter_grads)
}

/// Computes a vector-Jacobian product.
///
/// `f` maps a parameter tree on `B` to an output tree on `B`. The returned pullback maps a
/// cotangent tree on `B` to a parameter-gradient tree on `B`.
///
/// The pullback is repeatable: it re-runs `f` for each call, because Burn consumes the autodiff
/// tape on each backward pass. Use [`vjp_once`] when only one pullback call is needed and the
/// forward pass should run exactly once.
///
/// Multi-leaf outputs are combined into the scalar surrogate `sum_i sum(y_i * v_i)` before the
/// single backward pass, so shared intermediate nodes are handled correctly.
pub fn vjp<B, AB, F, Out, P>(f: F, primals: &P) -> (Out, impl Fn(&Out) -> P)
where
    B: Backend,
    AB: AutodiffBackend<InnerBackend = B>,
    P: PyTree<B, WithBackend<B> = P>,
    Out: PyTree<B, WithBackend<B> = Out>,
    F: Fn(&P::WithBackend<AB>) -> Out::WithBackend<AB>,
{
    let primal = primal_output::<B, AB, F, Out, P>(&f, primals);
    let primals = primals.clone();

    let pullback = move |cotangent: &Out| {
        let lifted = lift::<B, AB, P>(&primals);
        let out_ad = f(&lifted);
        pullback_from_out::<B, AB, Out, P>(&out_ad, &lifted, &primals, cotangent)
    };

    (primal, pullback)
}

/// Computes a vector-Jacobian product and returns a single-shot pullback.
///
/// Unlike [`vjp`], the pullback consumes the tape recorded by the forward pass and can only be
/// called once.
pub fn vjp_once<B, AB, F, Out, P>(f: F, primals: &P) -> (Out, impl FnOnce(&Out) -> P)
where
    B: Backend,
    AB: AutodiffBackend<InnerBackend = B>,
    P: PyTree<B, WithBackend<B> = P>,
    Out: PyTree<B, WithBackend<B> = Out>,
    F: Fn(&P::WithBackend<AB>) -> Out::WithBackend<AB>,
{
    let lifted = lift::<B, AB, P>(primals);
    let out_ad = f(&lifted);
    let primal = out_ad.map_float::<B, _>(&mut |leaf: &FloatTensor<AB>| AB::inner(leaf.clone()));
    let primals = primals.clone();

    let pullback = move |cotangent: &Out| {
        pullback_from_out::<B, AB, Out, P>(&out_ad, &lifted, &primals, cotangent)
    };

    (primal, pullback)
}

fn primal_output<B, AB, F, Out, P>(f: &F, primals: &P) -> Out
where
    B: Backend,
    AB: AutodiffBackend<InnerBackend = B>,
    P: PyTree<B, WithBackend<B> = P>,
    Out: PyTree<B, WithBackend<B> = Out>,
    F: Fn(&P::WithBackend<AB>) -> Out::WithBackend<AB>,
{
    let lifted = lift::<B, AB, P>(primals);
    let out_ad = f(&lifted);
    out_ad.map_float::<B, _>(&mut |leaf: &FloatTensor<AB>| AB::inner(leaf.clone()))
}

fn pullback_from_out<B, AB, Out, P>(
    out_ad: &Out::WithBackend<AB>,
    lifted: &P::WithBackend<AB>,
    primals: &P,
    cotangent: &Out,
) -> P
where
    B: Backend,
    AB: AutodiffBackend<InnerBackend = B>,
    P: PyTree<B, WithBackend<B> = P>,
    Out: PyTree<B, WithBackend<B> = Out>,
{
    // Materialize the scalar surrogate `sum_i sum(y_i * v_i)` so a single backward pass handles
    // outputs that share intermediate nodes. The zip returns the cotangent leaves unchanged; the
    // scalar is accumulated in the closure, which avoids requiring the rebound output type to be
    // its own rebound type.
    let mut scalar: Option<FloatTensor<AB>> = None;
    let _: Out = out_ad.zip_float::<B, B, _>(
        cotangent,
        &mut |output: &FloatTensor<AB>, seed: &FloatTensor<B>| {
            assert_eq!(
                output.shape(),
                seed.shape(),
                "Cotangent shape {:?} must match output shape {:?}",
                seed.shape(),
                output.shape()
            );
            let product = AB::float_mul(output.clone(), AB::from_inner(seed.clone()));
            let sum = AB::float_sum(product);
            scalar = Some(match scalar.take() {
                Some(accumulated) => AB::float_add(accumulated, sum),
                None => sum,
            });
            seed.clone()
        },
    );

    let scalar = scalar.unwrap_or_else(|| {
        // No output leaves. The gradient is zero; seed a scalar from any input leaf so the
        // backward pass still returns zero gradients for every parameter.
        let mut zero = None;
        lifted.visit_float(&mut |leaf: &FloatTensor<AB>| {
            if zero.is_none() {
                zero = Some(AB::float_zeros(
                    Shape::new([1]),
                    &leaf.device(),
                    leaf.dtype().into(),
                ));
            }
        });
        zero.expect("vjp requires at least one input or output leaf")
    });

    if !AB::is_tracked(&scalar) {
        return zero_like::<B, P>(primals);
    }

    let gradients = AB::backward(scalar);

    primals.zip_float::<AB, B, _>(
        lifted,
        &mut |original: &FloatTensor<B>, lifted_leaf: &FloatTensor<AB>| {
            AB::grad(lifted_leaf, &gradients).unwrap_or_else(|| {
                B::float_zeros(
                    original.shape(),
                    &original.device(),
                    original.dtype().into(),
                )
            })
        },
    )
}

fn lift<B, AB, P>(params: &P) -> P::WithBackend<AB>
where
    B: Backend,
    AB: AutodiffBackend<InnerBackend = B>,
    P: PyTree<B, WithBackend<B> = P>,
{
    params.map_float::<AB, _>(&mut |primitive: &FloatTensor<B>| {
        AB::float_set_require_grad(AB::from_inner(primitive.clone()), true)
    })
}

fn zero_like<B, P>(params: &P) -> P
where
    B: Backend,
    P: PyTree<B, WithBackend<B> = P>,
{
    params.map_float::<B, _>(&mut |primitive: &FloatTensor<B>| {
        B::float_zeros(
            primitive.shape(),
            &primitive.device(),
            primitive.dtype().into(),
        )
    })
}

/// A scalar function that can be evaluated at any backend level.
///
/// Unlike a closure, this trait lets [`hvp`] evaluate the same function at `Autodiff<B>` and at
/// `Autodiff<ForwardDiff<B>>`, which is what forward-over-reverse Hessian-vector products
/// require.
pub trait Differentiable<B: Backend, P: PyTree<B>> {
    /// Evaluates the scalar function on the requested backend.
    fn eval<B2: Backend>(&self, x: &P::WithBackend<B2>) -> FuncTensor<B2, 1>
    where
        P::WithBackend<B2>: PyTree<B2>;
}

/// Computes the gradient and a Hessian-vector product.
///
/// The HVP uses forward-over-reverse: it differentiates the reverse-mode gradient with a
/// forward-mode tangent. Both the gradient and the HVP are computed in a single pass
/// and returned on the primal backend.
pub fn hvp<B, P, F>(f: F, primals: &P, tangents: &P) -> (P, P)
where
    B: Backend,
    P: PyTree<B, WithBackend<B> = P>,
    P::WithBackend<ForwardDiff<B>>: PyTree<
            ForwardDiff<B>,
            WithBackend<ForwardDiff<B>> = P::WithBackend<ForwardDiff<B>>,
            WithBackend<Autodiff<ForwardDiff<B>>> = P::WithBackend<Autodiff<ForwardDiff<B>>>,
        >,
    F: Differentiable<B, P>,
{
    value_and_jvp::<B, _, P, P>(
        move |x: &P::WithBackend<ForwardDiff<B>>| {
            let gradient_on_forward =
                grad::<ForwardDiff<B>, _, P::WithBackend<ForwardDiff<B>>>(|y| f.eval(y));
            gradient_on_forward(x)
        },
        primals,
        tangents,
    )
}

/// Computes the gradient and a Hessian-vector product using a closure.
///
/// Both the gradient and the HVP are computed in a single forward-over-reverse pass
/// without requiring the [`Differentiable`] trait.
pub fn hvp_fn<B, P, F>(f: F, primals: &P, tangents: &P) -> (P, P)
where
    B: Backend,
    P: PyTree<B, WithBackend<B> = P>,
    P::WithBackend<ForwardDiff<B>>: PyTree<
            ForwardDiff<B>,
            WithBackend<ForwardDiff<B>> = P::WithBackend<ForwardDiff<B>>,
            WithBackend<Autodiff<ForwardDiff<B>>> = P::WithBackend<Autodiff<ForwardDiff<B>>>,
        >,
    F: Fn(&P::WithBackend<Autodiff<ForwardDiff<B>>>) -> FuncTensor<Autodiff<ForwardDiff<B>>, 1>,
{
    value_and_jvp::<B, _, P, P>(
        move |x: &P::WithBackend<ForwardDiff<B>>| {
            let gradient_on_forward = grad::<ForwardDiff<B>, _, P::WithBackend<ForwardDiff<B>>>(&f);
            gradient_on_forward(x)
        },
        primals,
        tangents,
    )
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use burn_autodiff::Autodiff;
    use burn_backend::{AutodiffBackend, Backend, Scalar, TensorData};
    use burn_flex::{Flex, FlexDevice};

    use super::*;

    type B = Flex;
    type AB = Autodiff<B>;

    fn tensor(values: &[f32]) -> FuncTensor<Flex, 1> {
        FuncTensor::from_data(TensorData::from(values), &FlexDevice)
    }

    fn values(tensor: FuncTensor<Flex, 1>) -> Vec<f32> {
        tensor.into_data_sync().try_to_vec::<f32>().unwrap()
    }

    #[test]
    fn grad_linear_function() {
        let device = FlexDevice;
        let x = FuncTensor::<B, 1>::from_data(TensorData::from([2.0f32]), &device);

        let f = |x: &FuncTensor<AB, 1>| (x.clone() * 3.0f64).sum();
        let dx = grad::<B, _, _>(f)(&x);

        let values = dx.into_data_sync().try_to_vec::<f32>().unwrap();
        assert_eq!(values, vec![3.0]);
    }

    #[test]
    fn grad_constant_function_is_zero() {
        let device = FlexDevice;
        let x = FuncTensor::<B, 1>::from_data(TensorData::from([2.0f32, 4.0f32]), &device);

        let f = |x: &FuncTensor<AB, 1>| {
            let _ = x;
            FuncTensor::<AB, 1>::ones([1], &FlexDevice)
        };
        let dx = grad::<B, _, _>(f)(&x);

        let values = dx.into_data_sync().try_to_vec::<f32>().unwrap();
        assert_eq!(values, vec![0.0, 0.0]);
    }

    #[derive(Clone)]
    struct Cubic;

    impl<B: Backend> Differentiable<B, FuncTensor<B, 1>> for Cubic {
        fn eval<B2: Backend>(&self, x: &FuncTensor<B2, 1>) -> FuncTensor<B2, 1> {
            (x.clone() * x.clone() * x.clone()).sum()
        }
    }

    #[test]
    fn hvp_of_cubic() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32]), &device);
        let v = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);

        let (gradient, hessian_vector) = hvp::<Flex, _, _>(Cubic, &x, &v);

        assert_eq!(
            gradient.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![12.0]
        );
        assert_eq!(
            hessian_vector.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![12.0]
        );
    }

    #[test]
    fn issue_6090_stateless_line_fit() {
        let device = FlexDevice;
        let x =
            FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 2.0, 3.0, 4.0]), &device);
        let y =
            FuncTensor::<Flex, 1>::from_data(TensorData::from([5.0f32, 7.0, 9.0, 11.0]), &device);
        let x_ad =
            FuncTensor::<AB, 1>::from_data(TensorData::from([1.0f32, 2.0, 3.0, 4.0]), &device);
        let y_ad =
            FuncTensor::<AB, 1>::from_data(TensorData::from([5.0f32, 7.0, 9.0, 11.0]), &device);

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

        let (_, pullback) =
            vjp::<Flex, AB, _, (FuncTensor<Flex, 1>, FuncTensor<Flex, 1>), _>(f, &x);

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
        let x = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]),
            &device,
        );
        let wrong_seed =
            FuncTensor::<Flex, 2>::from_data(TensorData::from([[1.0f32, 1.0]]), &device);
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
        let device = FlexDevice;
        let x =
            FuncTensor::<AB, 2>::from_data(TensorData::from([[0.2f32, -0.3], [0.5, 0.1]]), &device);
        let target = FuncTensor::<AB, 1>::from_data(TensorData::from([1.0f32, -1.0]), &device);

        let params = (
            FuncTensor::<Flex, 2>::from_data(
                TensorData::from([[0.5f32, -0.2], [0.1, 0.3]]),
                &device,
            ),
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

            let x_for_a1 =
                FuncTensor::<A1, 2>::from_primitive(A1::from_inner(x.primitive().clone()));

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

        let (gradient, hvp) = hvp_fn::<Flex, _, _>(
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
}
