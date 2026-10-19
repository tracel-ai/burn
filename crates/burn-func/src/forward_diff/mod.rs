use alloc::format;
use burn_backend::{
    Backend, BackendTypes, DType, DTypeUsageSet, ExecutionError, GraphUnsupported, MemoryPoolUsage,
    ProfileDuration, ProfileOptions, ProfileToken, Shape, SlicedPoolReport, TensorData,
    TensorMetadata, get_or_init_device_settings, tensor::FloatTensor,
};
use core::marker::PhantomData;

use crate::{PyTree, tensor::FuncTensor};

/// Module, integer, boolean and quantized delegation for `ForwardDiff`.
pub mod module;
/// Float tensor pushforward rules for `ForwardDiff`.
pub mod ops;

/// A primal value paired with an optional forward-mode tangent.
///
/// # Invariant
///
/// When `tangent` is `Some`, it has the same shape, dtype, and device as `primal`.
#[derive(Clone, Debug)]
pub struct DualFloatTensor<B: BackendTypes> {
    /// The primal value.
    pub primal: B::FloatTensorPrimitive,
    /// The tangent value.
    pub tangent: Option<B::FloatTensorPrimitive>,
}

impl<B: BackendTypes> DualFloatTensor<B> {
    /// Creates a dual with no tangent.
    pub fn constant(primal: B::FloatTensorPrimitive) -> Self {
        Self {
            primal,
            tangent: None,
        }
    }

    /// Creates a dual with a tangent.
    pub fn new(primal: B::FloatTensorPrimitive, tangent: B::FloatTensorPrimitive) -> Self {
        Self {
            primal,
            tangent: Some(tangent),
        }
    }
}

impl<B: BackendTypes> TensorMetadata for DualFloatTensor<B> {
    type Device = B::Device;

    fn dtype(&self) -> DType {
        self.primal.dtype()
    }

    fn shape(&self) -> Shape {
        self.primal.shape()
    }

    fn rank(&self) -> usize {
        self.primal.rank()
    }

    fn device(&self) -> Self::Device {
        self.primal.device()
    }

    fn can_mut(&self) -> bool {
        self.primal.can_mut()
    }
}

/// Forward-mode automatic differentiation as a backend decorator.
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardDiff<B> {
    _b: PhantomData<B>,
}

impl<B: Backend> BackendTypes for ForwardDiff<B> {
    type Device = B::Device;
    type FloatTensorPrimitive = DualFloatTensor<B>;
    type IntTensorPrimitive = B::IntTensorPrimitive;
    type BoolTensorPrimitive = B::BoolTensorPrimitive;
    type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive;
    type GraphPrimitive = GraphUnsupported;
}

impl<B: Backend> Backend for ForwardDiff<B> {
    fn ad_enabled(device: &Self::Device) -> bool {
        B::ad_enabled(device)
    }

    fn name(device: &Self::Device) -> alloc::string::String {
        format!("forward_diff<{}>", B::name(device))
    }

    fn seed(device: &Self::Device, seed: u64) {
        B::seed(device, seed)
    }

    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        B::sync(device)
    }

    fn profile<O: Send + 'static>(
        device: &Self::Device,
        options: ProfileOptions,
        func: impl FnOnce() -> O + Send,
    ) -> Result<(O, ProfileDuration), ExecutionError> {
        B::profile(device, options, func)
    }

    fn profile_start(device: &Self::Device) -> Result<Option<ProfileToken>, ExecutionError> {
        B::profile_start(device)
    }

    fn profile_end(
        device: &Self::Device,
        token: ProfileToken,
        options: ProfileOptions,
    ) -> Result<ProfileDuration, ExecutionError> {
        B::profile_end(device, token, options)
    }

    fn profile_abandon(device: &Self::Device, token: ProfileToken) {
        B::profile_abandon(device, token)
    }

    fn memory_persistent_allocations<
        Output: Send,
        Input: Send,
        Func: Fn(Input) -> Output + Send,
    >(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        B::memory_persistent_allocations(device, input, func)
    }

    fn memory_cleanup(device: &Self::Device) {
        B::memory_cleanup(device)
    }

    fn memory_pool_report(device: &Self::Device) -> Option<alloc::vec::Vec<SlicedPoolReport>> {
        B::memory_pool_report(device)
    }

    fn memory_pool_usage(device: &Self::Device) -> Option<MemoryPoolUsage> {
        B::memory_pool_usage(device)
    }

    fn staging<'a, Iter>(data: Iter, device: &Self::Device)
    where
        Iter: Iterator<Item = &'a mut TensorData>,
    {
        B::staging(data, device);
    }

    fn supports_dtype(device: &Self::Device, dtype: DType) -> bool {
        B::supports_dtype(device, dtype)
    }

    fn dtype_usage(device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        B::dtype_usage(device, dtype)
    }

    fn device_count(type_id: u16) -> usize {
        B::device_count(type_id)
    }

    fn flush(device: &Self::Device) -> Result<(), ExecutionError> {
        B::flush(device)
    }
}

/// Expands a tangent to the target shape when broadcasting changed the primal shape.
pub fn align_tangent<B: Backend>(tangent: FloatTensor<B>, target_shape: &Shape) -> FloatTensor<B> {
    let t_shape = tangent.shape();
    if t_shape == *target_shape {
        return tangent;
    }
    let tangent = if t_shape.num_dims() < target_shape.num_dims() {
        let mut new_dims: alloc::vec::Vec<usize> =
            alloc::vec![1; target_shape.num_dims() - t_shape.num_dims()];
        new_dims.extend(t_shape.iter().copied());
        B::float_reshape(tangent, Shape::from(new_dims))
    } else {
        tangent
    };
    B::float_expand(tangent, target_shape.clone())
}

/// Returns the device's integer dtype.
pub(crate) fn int_dtype<B: Backend>(device: &B::Device) -> burn_backend::IntDType {
    get_or_init_device_settings::<B>(device).int_dtype
}

/// Returns the device's boolean dtype.
pub(crate) fn bool_dtype<B: Backend>(device: &B::Device) -> burn_backend::BoolDType {
    get_or_init_device_settings::<B>(device).bool_dtype
}

/// Computes both the primal output and the forward-mode tangent.
///
/// `primals` and `tangents` must have identical structure and leaf shapes.
pub fn value_and_jvp<B, F, Out, P>(f: F, primals: &P, tangents: &P) -> (Out, Out)
where
    B: Backend,
    P: PyTree<B, WithBackend<B> = P>,
    Out: PyTree<B, WithBackend<B> = Out>,
    F: Fn(&P::WithBackend<ForwardDiff<B>>) -> Out::WithBackend<ForwardDiff<B>>,
{
    let duals = primals.zip_float::<B, ForwardDiff<B>, _>(tangents, &mut |primal, tangent| {
        assert_eq!(
            primal.shape(),
            tangent.shape(),
            "jvp primals and tangents must have identical leaf shapes"
        );
        DualFloatTensor::new(primal.clone(), tangent.clone())
    });

    let out_dual = f(&duals);

    let primal = out_dual.map_float::<B, _>(&mut |leaf: &DualFloatTensor<B>| leaf.primal.clone());
    let tangent = out_dual.map_float::<B, _>(&mut |leaf: &DualFloatTensor<B>| {
        leaf.tangent.clone().unwrap_or_else(|| {
            B::float_zeros(
                leaf.primal.shape(),
                &leaf.primal.device(),
                leaf.primal.dtype().into(),
            )
        })
    });

    (primal, tangent)
}

/// Computes the forward-mode Jacobian-vector product.
///
/// Returns the tangent of the output tree.
pub fn jvp<B, F, Out, P>(f: F, primals: &P, tangents: &P) -> Out
where
    B: Backend,
    P: PyTree<B, WithBackend<B> = P>,
    Out: PyTree<B, WithBackend<B> = Out>,
    F: Fn(&P::WithBackend<ForwardDiff<B>>) -> Out::WithBackend<ForwardDiff<B>>,
{
    value_and_jvp::<B, F, Out, P>(f, primals, tangents).1
}

/// A [`FuncTensor`] leaf on the `ForwardDiff` backend.
pub type ForwardTensor<B, const D: usize = 1> = FuncTensor<ForwardDiff<B>, D>;
