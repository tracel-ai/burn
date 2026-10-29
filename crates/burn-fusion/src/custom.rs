//! Metadata and support for opaque, deferred backend extensions.
//!
//! Metadata callbacks describe outputs with [`TensorSpec`](crate::custom::TensorSpec) before
//! registration. The generated wrapper assigns tensor IDs and queues an
//! [`Operation`](crate::stream::Operation) that invokes the inner backend and publishes its output
//! handles. Execution checks output enum variants in all builds and output dtypes and devices in
//! debug builds. Output shapes and ordinary field values are not checked against backend results.
//! Hidden items support macro expansions in downstream crates.
pub use crate::stream::{Operation, StreamId};
use crate::{ExecutionError, FusionBackend, FusionRuntime, FusionTensor};
#[cfg(debug_assertions)]
use burn_backend::Device;
use burn_backend::TensorMetadata;
pub use burn_ir::{
    BackendIr, CustomOpIr, HandleContainer, OperationIr, OperationOutput, ScalarIr, TensorHandle,
    TensorId, TensorIr,
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
/// Metadata contains the actual return values for non-tensor fields. These are returned without
/// waiting for execution and must equal what a direct backend call would return. The backend's values for
/// these fields are discarded without comparison.
#[doc(hidden)]
pub trait FusionValueAdapter<B: FusionBackend> {
    /// Tensor specs, enum variants, and ordinary return values supplied by the metadata callback.
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

/// Implementation support for custom operations that travel as IR alone, as they do to the remote
/// backend: the client builds a [`CustomOpIr`] carrying every argument, and the server rebuilds
/// the call from it.
///
/// Implemented for the tensor leaves and for derived extension values opting into `remote`. All
/// methods traverse tensor leaves in the same order as [`FusionValueAdapter`]. The non-tensor part
/// of a value (its enum variant and its ordinary fields) travels as scalars: the variant's index
/// first, then each field in declaration order, nested values inline.
#[doc(hidden)]
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot travel as a custom operation's IR",
    note = "add `remote` to the type's `#[extension_type(fusion, remote)]` attribute"
)]
pub trait RemoteValueAdapter<B: BackendIr> {
    /// The same metadata as [`FusionValueAdapter::Metadata`].
    type Metadata: Clone;
    /// The value holding tensor primitives of `B`.
    type Value;
    /// Describe a value without consuming it.
    fn to_metadata(value: &Self::Value) -> Self::Metadata;
    /// Encode the variant and ordinary fields, in the order [`read_input`](Self::read_input) decodes them.
    fn append_scalars(meta: &Self::Metadata, scalars: &mut Vec<ScalarIr>);
    /// Hand every tensor leaf to `input`, in leaf order.
    fn append_input(value: Self::Value, input: &mut impl FnMut(B::Handle));
    /// Append output IR for every tensor leaf the metadata describes.
    fn append_output_ir(
        meta: &Self::Metadata,
        out: &mut Vec<TensorIr>,
        create: &mut impl FnMut() -> TensorId,
    );
    /// Reassemble an output from the handles registration returned, taking the variant and
    /// ordinary fields from the metadata.
    fn build_output(
        meta: &Self::Metadata,
        handles: &mut impl Iterator<Item = B::Handle>,
    ) -> Self::Value;
    /// Rebuild an input from the operation's input tensors and scalars.
    fn read_input(
        handles: &mut HandleContainer<B::Handle>,
        inputs: &mut core::slice::Iter<'_, TensorIr>,
        scalars: &mut core::slice::Iter<'_, ScalarIr>,
    ) -> Self::Value;
    /// Register every tensor leaf of an output under the next output id.
    ///
    /// # Panics
    ///
    /// If the value has more tensors than the IR has outputs and, in debug builds, if a dtype
    /// differs from the one the client described.
    fn register_output(
        value: Self::Value,
        outputs: &mut core::slice::Iter<'_, TensorIr>,
        handles: &mut HandleContainer<B::Handle>,
    );
}

macro_rules! remote_tensor_adapter {
    ($name:ident, $primitive:ident, $to_handle:ident, $from_handle:ident, $register:ident, $get:ident, $check:expr) => {
        impl<B: BackendIr> RemoteValueAdapter<B> for $name {
            type Metadata = TensorSpec;
            type Value = B::$primitive;
            fn to_metadata(value: &Self::Value) -> TensorSpec { TensorSpec::new(value.shape(), value.dtype()) }
            fn append_scalars(_: &TensorSpec, _: &mut Vec<ScalarIr>) {}
            fn append_input(value: Self::Value, input: &mut impl FnMut(B::Handle)) { input(B::$to_handle(value)); }
            fn append_output_ir(meta: &TensorSpec, out: &mut Vec<TensorIr>, create: &mut impl FnMut() -> TensorId) {
                debug_assert!(($check)(meta.dtype), "custom output metadata has the wrong dtype category for {}", stringify!($name));
                out.push(TensorIr::uninit(create(), meta.shape.clone(), meta.dtype));
            }
            fn build_output(meta: &TensorSpec, handles: &mut impl Iterator<Item = B::Handle>) -> Self::Value {
                let handle = handles.next().expect("registration returns one handle per output");
                B::$from_handle(TensorHandle { handle, shape: meta.shape.clone() })
            }
            fn read_input(handles: &mut HandleContainer<B::Handle>, inputs: &mut core::slice::Iter<'_, TensorIr>, _: &mut core::slice::Iter<'_, ScalarIr>) -> Self::Value {
                handles.$get::<B>(inputs.next().expect("the custom operation's IR has fewer inputs than its signature"))
            }
            fn register_output(value: Self::Value, outputs: &mut core::slice::Iter<'_, TensorIr>, handles: &mut HandleContainer<B::Handle>) {
                let output = outputs.next().expect("the backend returned more output tensors than the custom operation's IR describes");
                debug_assert_eq!(value.dtype(), output.dtype, "the backend's output dtype differs from the custom operation's metadata");
                handles.$register::<B>(&output.id, value);
            }
        }
    };
}
remote_tensor_adapter!(
    Float,
    FloatTensorPrimitive,
    float_tensor_handle,
    float_tensor,
    register_float_tensor,
    get_float_tensor,
    |d: DType| d.is_float()
);
remote_tensor_adapter!(
    Int,
    IntTensorPrimitive,
    int_tensor_handle,
    int_tensor,
    register_int_tensor,
    get_int_tensor,
    |d: DType| d.is_int() || d.is_uint()
);
remote_tensor_adapter!(
    Bool,
    BoolTensorPrimitive,
    bool_tensor_handle,
    bool_tensor,
    register_bool_tensor,
    get_bool_tensor,
    |d: DType| d.is_bool()
);
remote_tensor_adapter!(
    Quantized,
    QuantizedTensorPrimitive,
    quantized_tensor_handle,
    quantized_tensor,
    register_quantized_tensor,
    get_quantized_tensor,
    |d: DType| matches!(d, DType::QFloat(_))
);

/// A primitive rebuilt from the scalar it travelled as in a custom operation's IR.
///
/// `usize` and `isize` travel as 64-bit integers; decoding one that does not fit the receiving
/// host's pointer width panics rather than truncating it.
#[doc(hidden)]
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no scalar decoding",
    note = "give the argument or field `decode = <callable>` beside its `scalar` encoding"
)]
pub trait FromScalarIr {
    /// Convert the scalar back into the value it encoded.
    fn from_scalar_ir(scalar: ScalarIr) -> Self;
}

macro_rules! from_scalar_ir {
    ($($ty:ty),*; $($wide:ty => $via:ty),*) => {
        $(impl FromScalarIr for $ty {
            fn from_scalar_ir(scalar: ScalarIr) -> Self {
                scalar.elem::<$ty>()
            }
        })*
        $(impl FromScalarIr for $wide {
            fn from_scalar_ir(scalar: ScalarIr) -> Self {
                <$wide>::try_from(scalar.elem::<$via>()).unwrap_or_else(|_| {
                    panic!("a `{}` scalar does not fit this host's `{}`", stringify!($via), stringify!($wide))
                })
            }
        })*
    };
}
from_scalar_ir!(
    bool, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64;
    isize => i64, usize => u64
);

/// The next scalar of a custom operation's IR.
///
/// # Panics
///
/// If the IR has fewer scalars than the operation's signature encodes.
#[doc(hidden)]
pub fn next_scalar(scalars: &mut core::slice::Iter<'_, ScalarIr>) -> ScalarIr {
    *scalars
        .next()
        .expect("the custom operation's IR has fewer scalars than its signature")
}

/// Decode `scalar` as the primitive `decode` takes, then hand it to `decode`.
#[doc(hidden)]
pub fn decode_scalar<T: FromScalarIr, R>(scalar: ScalarIr, decode: impl FnOnce(T) -> R) -> R {
    decode(T::from_scalar_ir(scalar))
}
