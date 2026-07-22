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
