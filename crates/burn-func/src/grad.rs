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
