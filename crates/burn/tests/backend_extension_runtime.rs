//! Runtime (executing) tests for enum/struct backend extension inputs, using the CPU `NdArray`
//! backend through `Dispatch`.
//!
//! The `backend_extension_remote` tests are compile-only (stubbed impls, function-pointer coercions).
//! These actually run the generated dispatch glue end to end: the runtime backend-selection walk,
//! the tensor-less enum variant panic, and enum-variant-dependent unwrapping.
#![cfg(feature = "ndarray")]

use burn::backend::{Dispatch, ExtensionType, NdArray, backend_extension, tensor::FloatTensor};
use burn::tensor::{Device, Tensor};

#[derive(ExtensionType)]
pub enum Operand<B: burn::backend::Backend> {
    Dense(FloatTensor<B>),
    Empty,
}

#[backend_extension(NdArray)]
pub trait RtBackend: burn::backend::Backend {
    /// Returns the active variant's tensor. Exercises enum-variant-dependent unwrapping.
    fn pick(#[extension_type] op: Operand<Self>) -> FloatTensor<Self>;
    /// Returns the bare tensor. When `op` is `Empty` the backend must be selected from `x`.
    fn mix_pick(x: FloatTensor<Self>, #[extension_type] op: Operand<Self>) -> FloatTensor<Self>;
}

impl RtBackend for NdArray {
    fn pick(op: Operand<Self>) -> FloatTensor<Self> {
        match op {
            Operand::Dense(x) => x,
            Operand::Empty => {
                unreachable!("Empty carries no tensor; the dispatch walk panics before calling")
            }
        }
    }

    fn mix_pick(x: FloatTensor<Self>, _op: Operand<Self>) -> FloatTensor<Self> {
        x
    }
}

fn device() -> Device {
    Device::ndarray()
}

#[test]
fn enum_input_runs_on_selected_backend() {
    let d = device();
    let t = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &d);
    let expected = t.clone().into_data();

    let out = <Dispatch as RtBackend>::pick(Operand::Dense(t.into_dispatch()));
    let out = Tensor::<1>::from_dispatch(out);

    out.into_data().assert_eq(&expected, true);
}

#[test]
fn mixed_input_selects_backend_from_bare_tensor_when_enum_is_tensorless() {
    let d = device();
    let x = Tensor::<1>::from_floats([4.0, 5.0], &d);
    let expected = x.clone().into_data();

    // The enum is `Empty` (no tensor), so the walk must select the backend from `x`.
    let out = <Dispatch as RtBackend>::mix_pick(x.into_dispatch(), Operand::Empty);
    let out = Tensor::<1>::from_dispatch(out);

    out.into_data().assert_eq(&expected, true);
}

#[test]
#[should_panic(expected = "no tensor input to select a backend from")]
fn all_tensorless_input_panics() {
    // The only input is a tensor-less enum variant, so the backend is unresolvable: the walk's
    // `.expect(...)` fires. `pick`'s `NdArray` impl is never reached.
    let _ = <Dispatch as RtBackend>::pick(Operand::Empty);
}

// End-to-end autodiff: a differentiable op over a struct input, with a hand-written `Backward`, run
// on `NdArray` through `Dispatch`. Verifies gradients actually flow back into the struct's fields
// (not just that the dispatch glue type-checks).
#[cfg(feature = "autodiff")]
mod autodiff_gradients {
    use super::*;
    use burn::backend::Backend;
    use burn::backend::autodiff::{
        Autodiff,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    };
    use burn::backend::ops::FloatTensorOps;
    use burn::tensor::TensorData;

    #[derive(ExtensionType)]
    pub struct FloatPair<B: burn::backend::Backend> {
        pub x: FloatTensor<B>,
        pub y: FloatTensor<B>,
    }

    #[backend_extension(Autodiff, NdArray)]
    pub trait GradBackend: burn::backend::Backend {
        /// Elementwise `x * y`, differentiable in both fields.
        fn mul_pair(#[extension_type] p: FloatPair<Self>) -> FloatTensor<Self>;
    }

    // Concrete forward.
    impl GradBackend for NdArray {
        fn mul_pair(p: FloatPair<Self>) -> FloatTensor<Self> {
            NdArray::float_mul(p.x, p.y)
        }
    }

    // Autodiff: register the backward step over the struct's two tracked float fields.
    impl<C: CheckpointStrategy> GradBackend for Autodiff<NdArray, C> {
        fn mul_pair(p: FloatPair<Self>) -> FloatTensor<Self> {
            #[derive(Debug)]
            struct MulPairBackward;

            impl<B: Backend> Backward<B, 2> for MulPairBackward {
                // d(x*y): grad_x = grad * y, grad_y = grad * x. Save the forward inputs.
                type State = (FloatTensor<B>, FloatTensor<B>);

                fn backward(
                    self,
                    ops: Ops<Self::State, 2>,
                    grads: &mut Gradients,
                    _checkpointer: &mut Checkpointer,
                ) {
                    let [node_x, node_y] = ops.parents;
                    let grad = grads.consume::<B>(&ops.node);
                    let (x, y) = ops.state;

                    if let Some(node) = node_x {
                        grads.register::<B>(node.id, B::float_mul(grad.clone(), y));
                    }
                    if let Some(node) = node_y {
                        grads.register::<B>(node.id, B::float_mul(grad, x));
                    }
                }
            }

            match MulPairBackward
                .prepare::<C>([p.x.node.clone(), p.y.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(prep) => {
                    let x = p.x.primitive.clone();
                    let y = p.y.primitive.clone();
                    let output = NdArray::float_mul(x.clone(), y.clone());
                    prep.finish((x, y), output)
                }
                OpsKind::UnTracked(prep) => {
                    prep.finish(NdArray::float_mul(p.x.primitive, p.y.primitive))
                }
            }
        }
    }

    #[test]
    fn autodiff_struct_input_propagates_gradients() {
        let device = Device::ndarray().autodiff();
        let x = Tensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
        let y = Tensor::<1>::from_floats([4.0, 5.0], &device).require_grad();

        let out = Tensor::<1>::from_dispatch(<Dispatch as GradBackend>::mul_pair(FloatPair {
            x: x.clone().into_dispatch(),
            y: y.clone().into_dispatch(),
        }));

        // Forward: x * y.
        out.clone()
            .into_data()
            .assert_eq(&TensorData::from([8.0f32, 15.0]), true);

        let grads = out.backward();
        // grad_x = y, grad_y = x.
        x.grad(&grads)
            .unwrap()
            .into_data()
            .assert_eq(&TensorData::from([4.0f32, 5.0]), true);
        y.grad(&grads)
            .unwrap()
            .into_data()
            .assert_eq(&TensorData::from([2.0f32, 3.0]), true);
    }
}
