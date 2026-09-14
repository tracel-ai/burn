//! Metadata and support for opaque, deferred backend extensions.
//!
//! Metadata callbacks describe outputs with [`TensorSpec`](crate::custom::TensorSpec) before
//! registration. The generated wrapper assigns tensor IDs and queues an
//! [`Operation`](crate::stream::Operation) that invokes the inner backend, validates its outputs,
//! and publishes their handles. Hidden items support macro expansions in downstream crates.
pub use crate::stream::{Operation, StreamId};
use crate::{ExecutionError, FusionBackend, FusionRuntime, FusionTensor};
use burn_backend::{Device, TensorMetadata};
pub use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, ScalarIr, TensorIr};
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
/// association lets imported and renamed field types work without importing their companion types.
#[doc(hidden)]
pub trait ExtensionMetadata {
    /// The generated companion containing tensor specs, nested metadata, and ordinary fields.
    type Metadata: Clone + core::fmt::Debug;
}

/// Implementation support for tensor primitives and derived extension values.
///
/// All methods traverse tensor leaves in the same order: tuple position and field declaration
/// order, recursively. Registration flattens metadata into output IR and reconstructs lazy tensors.
/// Deferred execution validates every output before publishing any handles, so a mismatch fails
/// the entire operation rather than exposing a partially valid result.
#[doc(hidden)]
pub trait Layout<B: FusionBackend> {
    /// Shapes and dtypes supplied by the metadata callback.
    type Metadata: Clone;
    /// Output value produced by the inner backend `B`.
    type Base;
    /// Corresponding output value containing `Fusion<B>` tensor primitives.
    type Target;
    /// Describe inputs without retaining tensor handles.
    fn metadata(value: &Self::Target) -> Self::Metadata;
    /// Visit input tensors to select a client and check their devices.
    fn visit_tensors(value: &Self::Target, visit: &mut impl FnMut(&FusionTensor<B::FusionRuntime>));
    /// Consume inputs in declaration order, retaining Fusion's ownership information.
    fn into_inputs(value: Self::Target, inputs: &mut Vec<TensorIr>);
    /// Resolve input handles, restoring the variant and ordinary fields from metadata.
    fn read(
        meta: &Self::Metadata,
        inputs: &mut core::slice::Iter<'_, TensorIr>,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) -> Self::Base;
    /// Append output specs, rejecting dtypes outside each tensor's category before registration.
    fn specs(meta: &Self::Metadata, out: &mut Vec<TensorSpec>);
    /// Consume the expected IR entries and check actual shapes, dtypes, and devices.
    fn validate(
        value: &Self::Base,
        meta: &Self::Metadata,
        specs: &mut core::slice::Iter<'_, TensorIr>,
        device: &B::Device,
    ) -> Result<(), ExecutionError>;
    /// Publish validated handles, consuming the same IR entries from a fresh iterator.
    fn publish(
        value: Self::Base,
        specs: &mut core::slice::Iter<'_, TensorIr>,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    );
    /// Reassemble the output value from the lazy tensors returned by registration.
    fn reconstruct(
        meta: &Self::Metadata,
        tensors: &mut std::vec::IntoIter<FusionTensor<B::FusionRuntime>>,
    ) -> Self::Target;
}

macro_rules! tensor_layout {
    ($name:ident, $primitive:ident, $register:ident, $get:ident, $check:expr) => {
        #[doc(hidden)]
        pub struct $name;
        impl<B: FusionBackend> Layout<B> for $name {
            type Metadata = TensorSpec;
            type Base = B::$primitive;
            type Target = FusionTensor<B::FusionRuntime>;
            fn metadata(value: &Self::Target) -> TensorSpec { TensorSpec::new(value.shape.clone(), value.dtype) }
            fn visit_tensors(value: &Self::Target, visit: &mut impl FnMut(&FusionTensor<B::FusionRuntime>)) { visit(value); }
            fn into_inputs(value: Self::Target, inputs: &mut Vec<TensorIr>) { inputs.push(value.into_ir()); }
            fn read(_: &TensorSpec, inputs: &mut core::slice::Iter<'_, TensorIr>, handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>) -> Self::Base {
                handles.$get::<B>(inputs.next().expect("input layout"))
            }
            fn specs(meta: &TensorSpec, out: &mut Vec<TensorSpec>) {
                assert!(($check)(meta.dtype), "Fusion output metadata has the wrong dtype category for {}", stringify!($name));
                out.push(meta.clone());
            }
            fn validate(value: &Self::Base, _: &TensorSpec, specs: &mut core::slice::Iter<'_, TensorIr>, device: &B::Device) -> Result<(), ExecutionError> {
                let expected = specs.next().expect("output layout");
                if value.shape() != expected.shape || value.dtype() != expected.dtype || value.device().to_id() != device.to_id() {
                    return Err(ExecutionError::generic(format!("Fusion custom output metadata mismatch: expected {:?} {:?} on {:?}, got {:?} {:?} on {:?}", expected.shape, expected.dtype, device.to_id(), value.shape(), value.dtype(), value.device().to_id())));
                }
                Ok(())
            }
            fn publish(value: Self::Base, specs: &mut core::slice::Iter<'_, TensorIr>, handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>) {
                handles.$register::<B>(&specs.next().expect("output layout").id, value);
            }
            fn reconstruct(_: &TensorSpec, tensors: &mut std::vec::IntoIter<FusionTensor<B::FusionRuntime>>) -> Self::Target { tensors.next().expect("output layout") }
        }
    }
}
tensor_layout!(
    Float,
    FloatTensorPrimitive,
    register_float_tensor,
    get_float_tensor,
    |d: DType| d.is_float()
);
tensor_layout!(
    Int,
    IntTensorPrimitive,
    register_int_tensor,
    get_int_tensor,
    |d: DType| d.is_int() || d.is_uint()
);
tensor_layout!(
    Bool,
    BoolTensorPrimitive,
    register_bool_tensor,
    get_bool_tensor,
    |d: DType| d.is_bool()
);
tensor_layout!(
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
