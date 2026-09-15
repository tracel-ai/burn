//! Metadata and support for opaque, deferred backend extensions.
//!
//! Metadata callbacks describe outputs with [`TensorSpec`](crate::custom::TensorSpec) before
//! registration. The generated wrapper assigns tensor IDs and queues an
//! [`Operation`](crate::stream::Operation) that invokes the inner backend, validates its outputs,
//! and publishes their handles. Hidden items support macro expansions in downstream crates.
pub use crate::stream::{Operation, StreamId};
use crate::{ExecutionError, FusionBackend, FusionRuntime, FusionTensor};
#[cfg(debug_assertions)]
use burn_backend::{Device, TensorMetadata};
pub use burn_ir::{
    CustomOpIr, HandleContainer, OperationIr, OperationOutput, ScalarIr, TensorId, TensorIr,
};
use burn_std::{DType, Shape};

/// Shape and element type of a tensor, without access to its data or handle.
///
/// Metadata callbacks receive specs for inputs and return specs for outputs.
#[derive(Clone, Debug)]
pub struct TensorSpec {
    /// Tensor dimensions.
    pub shape: Shape,
    /// Tensor element type.
    pub dtype: DType,
}
impl TensorSpec {
    /// Describe an output tensor.
    pub fn new(shape: Shape, dtype: DType) -> Self {
        Self { shape, dtype }
    }
}

/// Associates an extension value with its backend-independent metadata.
///
/// Generated for structs and enums opting into Fusion support. Resolving nested metadata through this
/// association lets imported and renamed field types work without importing their generated metadata types.
#[doc(hidden)]
pub trait ExtensionMetadata {
    /// The generated metadata type containing tensor specs, nested metadata, and ordinary fields.
    type Metadata: Clone + core::fmt::Debug;
    /// Input descriptors and ordinary fields retained for execution.
    type Input;
    /// Output IDs and enum variants retained for execution.
    type Output;
}

/// Implementation support for tensor primitives and derived extension values.
///
/// All methods traverse tensor leaves in the same order: tuple position and field declaration
/// order, recursively. Registration flattens metadata into output IR and reconstructs lazy tensors.
/// Deferred execution checks output variants before publishing any handles.
/// Debug builds additionally check output dtypes and devices.
/// Fusion uses the metadata callback's output shapes without checking them against the backend results.
/// Ordinary output fields are reconstructed from metadata; the inner backend's values are ignored.
#[doc(hidden)]
pub trait FusionValueAdapter<B: FusionBackend> {
    /// Shapes and dtypes supplied by the metadata callback.
    type Metadata: Clone;
    /// Input descriptors and ordinary fields retained for execution.
    type Input;
    /// Output IDs and enum variants retained for execution.
    type Output;
    /// Value containing tensor primitives of the inner backend `B`.
    type Inner;
    /// Corresponding value containing `Fusion<B>` tensor primitives.
    type Fused;
    /// Describe inputs without retaining tensor handles.
    fn to_metadata(value: &Self::Fused) -> Self::Metadata;
    /// Visit input tensors to select a client and check their devices.
    fn visit_fused_tensors(
        value: &Self::Fused,
        visit: &mut impl FnMut(&FusionTensor<B::FusionRuntime>),
    );
    /// Number of tensor leaves, used to allocate the IR vectors once.
    fn tensor_count(meta: &Self::Metadata) -> usize;
    /// Consume inputs in declaration order and retain their execution state.
    fn append_input_ir(value: Self::Fused, inputs: &mut Vec<TensorIr>) -> Self::Input;
    /// Resolve input handles, restoring the variant and ordinary fields from execution state.
    fn resolve_inputs(
        input: &Self::Input,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) -> Self::Inner;
    /// Append output IR and retain only the state needed to publish results.
    fn append_output_ir(
        meta: &Self::Metadata,
        out: &mut Vec<TensorIr>,
        create: &mut impl FnMut() -> TensorId,
    ) -> Self::Output;
    /// Check variants, and optionally dtypes and devices, before publishing any handles.
    fn validate_outputs(
        value: &Self::Inner,
        output: &Self::Output,
        device: Option<&B::Device>,
    ) -> Result<(), ExecutionError>;
    /// Register validated output handles under their assigned IDs.
    fn register_output_handles(
        value: Self::Inner,
        output: &Self::Output,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    );
    /// Reassemble the output value from the lazy tensors returned by registration.
    fn build_fused_output(
        meta: &Self::Metadata,
        tensors: &mut std::vec::IntoIter<FusionTensor<B::FusionRuntime>>,
    ) -> Self::Fused;
}

/// Execution state for a tensor output; its shape lives in the operation IR.
#[doc(hidden)]
pub struct OutputTensor {
    id: TensorId,
    #[cfg(debug_assertions)]
    dtype: DType,
}

macro_rules! tensor_adapter {
    ($name:ident, $primitive:ident, $register:ident, $get:ident, $check:expr) => {
        #[doc(hidden)]
        pub struct $name;
        impl<B: FusionBackend> FusionValueAdapter<B> for $name {
            type Metadata = TensorSpec;
            type Input = TensorIr;
            type Output = OutputTensor;
            type Inner = B::$primitive;
            type Fused = FusionTensor<B::FusionRuntime>;
            fn to_metadata(value: &Self::Fused) -> TensorSpec { TensorSpec::new(value.shape.clone(), value.dtype) }
            fn visit_fused_tensors(value: &Self::Fused, visit: &mut impl FnMut(&FusionTensor<B::FusionRuntime>)) { visit(value); }
            fn tensor_count(_: &TensorSpec) -> usize { 1 }
            fn append_input_ir(value: Self::Fused, inputs: &mut Vec<TensorIr>) -> TensorIr {
                let input = value.into_ir();
                inputs.push(input.clone());
                input
            }
            fn resolve_inputs(input: &TensorIr, handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>) -> Self::Inner {
                handles.$get::<B>(input)
            }
            fn append_output_ir(meta: &TensorSpec, out: &mut Vec<TensorIr>, create: &mut impl FnMut() -> TensorId) -> OutputTensor {
                debug_assert!(($check)(meta.dtype), "Fusion output metadata has the wrong dtype category for {}", stringify!($name));
                let id = create();
                out.push(TensorIr::uninit(id, meta.shape.clone(), meta.dtype));
                OutputTensor { id, #[cfg(debug_assertions)] dtype: meta.dtype }
            }
            fn validate_outputs(_value: &Self::Inner, _output: &OutputTensor, _device: Option<&B::Device>) -> Result<(), ExecutionError> {
                #[cfg(debug_assertions)]
                if let Some(device) = _device {
                    if _value.dtype() != _output.dtype || _value.device().to_id() != device.to_id() {
                        return Err(ExecutionError::generic(format!("Fusion custom output metadata mismatch: expected {:?} on {:?}, got {:?} on {:?}", _output.dtype, device.to_id(), _value.dtype(), _value.device().to_id())));
                    }
                }
                Ok(())
            }
            fn register_output_handles(value: Self::Inner, output: &OutputTensor, handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>) {
                handles.$register::<B>(&output.id, value);
            }
            fn build_fused_output(_: &TensorSpec, tensors: &mut std::vec::IntoIter<FusionTensor<B::FusionRuntime>>) -> Self::Fused { tensors.next().expect("output layout") }
        }
    }
}
tensor_adapter!(
    Float,
    FloatTensorPrimitive,
    register_float_tensor,
    get_float_tensor,
    |d: DType| d.is_float()
);
tensor_adapter!(
    Int,
    IntTensorPrimitive,
    register_int_tensor,
    get_int_tensor,
    |d: DType| d.is_int() || d.is_uint()
);
tensor_adapter!(
    Bool,
    BoolTensorPrimitive,
    register_bool_tensor,
    get_bool_tensor,
    |d: DType| d.is_bool()
);
tensor_adapter!(
    Quantized,
    QuantizedTensorPrimitive,
    register_quantized_tensor,
    get_quantized_tensor,
    |d: DType| matches!(d, DType::QFloat(_))
);

/// Adapt a closure to an operation without imposing Debug on captured options.
#[doc(hidden)]
pub struct OperationFn<F>(pub F);
impl<F> core::fmt::Debug for OperationFn<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("BackendExtension")
    }
}
impl<R: FusionRuntime, F> Operation<R> for OperationFn<F>
where
    F: Fn(&mut HandleContainer<R::FusionHandle>) -> Result<(), ExecutionError> + Send + Sync,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) -> Result<(), ExecutionError> {
        (self.0)(handles)
    }
}
