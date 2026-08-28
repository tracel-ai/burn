//! Runtime (executing) tests for enum/struct backend extension inputs, using the CPU `NdArray`
//! backend through `Dispatch`.
//!
//! The `backend_extension_remote` tests are compile-only (stubbed impls, function-pointer coercions).
//! These actually run the generated dispatch glue end to end: the runtime backend-selection walk,
//! the tensor-less enum variant panic, and enum-variant-dependent unwrapping.
#![cfg(feature = "ndarray")]
// `multi_backend_autodiff` needs two concrete backends compiled in at once to prove the dispatch
// walk inspects the backend inside `DispatchTensorKind::Autodiff` rather than letting the first
// generated arm capture everything. This suite still pairs NdArray with Flex (the latter coming
// from the default features); `Cpu` + Flex would work without a GPU or libtorch too, so this
// should move to that pair before burn-ndarray is removed.
#![allow(deprecated)]

use burn::backend::{
    Dispatch, ExtensionType, FloatDType, NdArray, backend_extension,
    ops::IntTensorOps,
    tensor::{FloatTensor, IntTensor},
};
use burn::tensor::{Device, Int, Tensor};

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

#[cfg(feature = "autodiff")]
mod associated_integer_output {
    use super::*;
    use burn::backend::Autodiff;
    use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;

    #[backend_extension(Autodiff, NdArray)]
    trait AssociatedBackend: burn::backend::Backend {
        fn int_to_float(x: IntTensor<Self>) -> FloatTensor<Self>;
    }

    impl AssociatedBackend for NdArray {
        fn int_to_float(x: IntTensor<Self>) -> FloatTensor<Self> {
            Self::int_into_float(x, FloatDType::F32)
        }
    }

    // Required for the statically generated float-input route, even though this operation has no
    // float input and therefore executes the concrete implementation at runtime.
    impl<C: CheckpointStrategy> AssociatedBackend for Autodiff<NdArray, C> {
        fn int_to_float(x: IntTensor<Self>) -> FloatTensor<Self> {
            Self::int_into_float(x, FloatDType::F32)
        }
    }

    #[test]
    fn associated_integer_to_float_lifts_the_concrete_result() {
        let device = Device::ndarray().autodiff();
        let input = Tensor::<1, Int>::from_data([1, 2, 3], &device);
        let output = <Dispatch as AssociatedBackend>::int_to_float(input.into_dispatch());
        let output = Tensor::<1>::from_dispatch(output);

        // Calling device used to panic because the extension attached enabled metadata to a
        // concrete float primitive instead of lifting it into the selected autodiff backend.
        assert_eq!(output.device(), device);
    }
}

// Regression test for autodiff dispatch with more than one concrete backend. Float dispatch tensors
// store the concrete backend inside `DispatchTensorKind::Autodiff`, so routing must inspect that
// inner kind instead of allowing the first generated autodiff arm to capture every backend.
#[cfg(all(feature = "autodiff", feature = "flex"))]
mod multi_backend_autodiff {
    use super::*;
    use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
    use burn::backend::{Autodiff, Backend, Flex};

    #[backend_extension(NdArray, Flex, Autodiff)]
    pub trait DoubleBackend: Backend {
        fn double(x: FloatTensor<Self>) -> FloatTensor<Self> {
            Self::float_add(x.clone(), x)
        }
    }

    impl DoubleBackend for NdArray {}
    impl DoubleBackend for Flex {}
    impl<B: Backend + DoubleBackend, C: CheckpointStrategy> DoubleBackend for Autodiff<B, C> {}

    fn assert_double(device: Device) {
        let x = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device);
        let out = <Dispatch as DoubleBackend>::double(x.into_dispatch());
        Tensor::<1>::from_dispatch(out)
            .into_data()
            .assert_eq(&burn::tensor::TensorData::from([2.0f32, 4.0, 6.0]), true);
    }

    #[test]
    fn autodiff_float_input_dispatches_to_each_concrete_backend() {
        assert_double(Device::ndarray().autodiff());
        assert_double(Device::flex().autodiff());
    }
}

#[cfg(feature = "autodiff")]
mod checkpoint_strategy_routing {
    use super::*;
    use burn::backend::autodiff::checkpoint::strategy::{
        BalancedCheckpointing, CheckpointStrategy,
    };
    use burn::backend::ops::FloatTensorOps;
    use burn::backend::{Autodiff, Backend, GradientCheckpointingStrategy};
    use core::any::TypeId;
    use core::sync::atomic::{AtomicU8, Ordering};

    static EXECUTED_STRATEGY: AtomicU8 = AtomicU8::new(0);

    #[backend_extension(Autodiff, NdArray)]
    trait StrategyBackend: Backend {
        fn identity(x: FloatTensor<Self>) -> FloatTensor<Self>;
        fn add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self>;
    }

    impl StrategyBackend for NdArray {
        fn identity(x: FloatTensor<Self>) -> FloatTensor<Self> {
            x
        }

        fn add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
            Self::float_add(lhs, rhs)
        }
    }

    impl<C: CheckpointStrategy> StrategyBackend for Autodiff<NdArray, C> {
        fn identity(x: FloatTensor<Self>) -> FloatTensor<Self> {
            let selected = if TypeId::of::<C>() == TypeId::of::<BalancedCheckpointing>() {
                2
            } else {
                1
            };
            EXECUTED_STRATEGY.store(selected, Ordering::SeqCst);
            x
        }

        fn add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
            Self::float_add(lhs, rhs)
        }
    }

    #[test]
    fn executes_the_strategy_carried_by_dispatch_metadata() {
        for (strategy, expected) in [
            (GradientCheckpointingStrategy::Disabled, 1),
            (GradientCheckpointingStrategy::Balanced, 2),
        ] {
            let device = Device::ndarray().autodiff();
            let input = Tensor::<1>::from_floats([1.0], &device)
                .with_gradient_checkpointing_strategy(strategy);
            let output = <Dispatch as StrategyBackend>::identity(input.into_dispatch());
            let output = Tensor::<1>::from_dispatch(output);

            assert_eq!(EXECUTED_STRATEGY.load(Ordering::SeqCst), expected);
            assert_eq!(output.device().gradient_checkpointing_strategy(), strategy);
        }
    }

    #[test]
    fn uniform_autodiff_floats_preserve_their_strategy() {
        let strategy = GradientCheckpointingStrategy::Balanced;
        let lhs = Tensor::<1>::from_floats([1.0], &Device::ndarray().autodiff())
            .with_gradient_checkpointing_strategy(strategy);
        let rhs = Tensor::<1>::from_floats([2.0], &Device::ndarray().autodiff())
            .with_gradient_checkpointing_strategy(strategy);

        let output = <Dispatch as StrategyBackend>::add(lhs.into_dispatch(), rhs.into_dispatch());
        let output = Tensor::<1>::from_dispatch(output);
        assert_eq!(output.device().gradient_checkpointing_strategy(), strategy);
        output
            .into_data()
            .assert_eq(&burn::tensor::TensorData::from([3.0f32]), true);
    }
}

#[cfg(feature = "autodiff")]
mod extension_context_contract {
    use super::*;
    use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
    use burn::backend::{Autodiff, GradientCheckpointingStrategy};
    use core::sync::atomic::{AtomicBool, Ordering};

    static IMPLEMENTATION_CALLED: AtomicBool = AtomicBool::new(false);

    #[derive(ExtensionType)]
    struct IntPair<B: burn::backend::Backend> {
        left: IntTensor<B>,
        right: IntTensor<B>,
    }

    #[derive(ExtensionType)]
    enum IntChoice<B: burn::backend::Backend> {
        Dense(IntTensor<B>),
        Empty,
    }

    #[derive(ExtensionType)]
    struct NestedInputs<B: burn::backend::Backend> {
        #[extension_type]
        pair: IntPair<B>,
        #[extension_type]
        choice: IntChoice<B>,
    }

    #[backend_extension(Autodiff, NdArray)]
    trait ContextBackend: burn::backend::Backend {
        fn select(#[extension_type] inputs: NestedInputs<Self>) -> IntTensor<Self>;
        fn select_conflict(#[extension_type] inputs: NestedInputs<Self>) -> IntTensor<Self>;
    }

    impl ContextBackend for NdArray {
        fn select(inputs: NestedInputs<Self>) -> IntTensor<Self> {
            inputs.pair.left
        }

        fn select_conflict(inputs: NestedInputs<Self>) -> IntTensor<Self> {
            IMPLEMENTATION_CALLED.store(true, Ordering::SeqCst);
            inputs.pair.left
        }
    }

    impl<C: CheckpointStrategy> ContextBackend for Autodiff<NdArray, C> {
        fn select(inputs: NestedInputs<Self>) -> IntTensor<Self> {
            inputs.pair.left
        }

        fn select_conflict(inputs: NestedInputs<Self>) -> IntTensor<Self> {
            IMPLEMENTATION_CALLED.store(true, Ordering::SeqCst);
            inputs.pair.left
        }
    }

    fn int(strategy: GradientCheckpointingStrategy) -> Tensor<1, Int> {
        Tensor::from_data([1], &Device::ndarray().autodiff())
            .with_gradient_checkpointing_strategy(strategy)
    }

    #[test]
    fn routing_context_propagates_through_nested_fields() {
        let balanced = int(GradientCheckpointingStrategy::Balanced);
        let output = <Dispatch as ContextBackend>::select(NestedInputs {
            pair: IntPair {
                left: balanced.clone().into_dispatch(),
                right: balanced.clone().into_dispatch(),
            },
            choice: IntChoice::Dense(balanced.into_dispatch()),
        });
        let output = Tensor::<1, Int>::from_dispatch(output);

        assert_eq!(
            output.device().gradient_checkpointing_strategy(),
            GradientCheckpointingStrategy::Balanced
        );
    }

    #[test]
    fn invalid_nested_tensor_representation_panics_before_the_implementation() {
        IMPLEMENTATION_CALLED.store(false, Ordering::SeqCst);
        // `IntPair` declares an integer field, but dispatch primitives share one runtime type, so a
        // malformed downstream value can place an autodiff float representation in that field.
        let malformed = Tensor::<1>::from_data([1.0], &Device::ndarray().autodiff())
            .with_gradient_checkpointing_strategy(GradientCheckpointingStrategy::Balanced)
            .into_dispatch();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            <Dispatch as ContextBackend>::select_conflict(NestedInputs {
                pair: IntPair {
                    left: malformed,
                    right: int(GradientCheckpointingStrategy::Balanced).into_dispatch(),
                },
                choice: IntChoice::Empty,
            })
        }));

        assert!(result.is_err());
        assert!(!IMPLEMENTATION_CALLED.load(Ordering::SeqCst));
    }
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
