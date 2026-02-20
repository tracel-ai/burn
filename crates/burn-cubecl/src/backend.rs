use crate::{
    CubeRuntime, FloatElement, IntElement,
    element::BoolElement,
    tensor::{CubeInplaceTensor, CubeTensor},
};
use burn_backend::{Backend, DTypeUsage, DTypeUsageSet, DeviceOps, ExecutionError, TensorData};
use burn_std::DType;
use cubecl::{
    features::{MmaConfig, TypeUsage},
    server::ComputeServer,
};
use std::marker::PhantomData;

#[cfg(not(feature = "fusion"))]
use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
#[cfg(not(feature = "fusion"))]
use burn_ir::{BackendIr, TensorHandle};

/// Generic tensor backend that can be compiled just-in-time to any shader runtime
#[derive(new)]
pub struct CubeBackend<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> {
    _runtime: PhantomData<R>,
    _float_elem: PhantomData<F>,
    _int_elem: PhantomData<I>,
    _bool_elem: PhantomData<BT>,
}

impl<R, F, I, BT> Backend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    type Device = R::Device;

    type FloatElem = F;
    type IntElem = I;
    type BoolElem = BT;

    type FloatTensorPrimitive = CubeTensor<R>;
    type IntTensorPrimitive = CubeTensor<R>;
    type BoolTensorPrimitive = CubeTensor<R>;
    type QuantizedTensorPrimitive = CubeTensor<R>;
    type CommunicationTensorPrimitive = CubeInplaceTensor<R>;

    fn name(device: &Self::Device) -> String {
        let client = R::client(device);
        format!("cubecl<{}>", R::name(&client))
    }

    fn seed(_device: &Self::Device, seed: u64) {
        cubek::random::seed(seed);
    }

    fn ad_enabled() -> bool {
        false
    }

    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = R::client(device);
        futures_lite::future::block_on(client.sync()).map_err(|err| ExecutionError::WithContext {
            reason: format!("{err}"),
        })
    }

    fn memory_persistent_allocations<Output, Input, Func: Fn(Input) -> Output>(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        let client = R::client(device);
        client.memory_persistent_allocation(input, func)
    }

    fn memory_cleanup(device: &Self::Device) {
        let client = R::client(device);
        client.memory_cleanup();
    }

    fn staging<'a, Iter>(data: Iter, device: &Self::Device)
    where
        Iter: Iterator<Item = &'a mut TensorData>,
    {
        let client = R::client(device);
        client.staging(data.map(|td| &mut td.bytes), false);
    }

    fn supports_dtype(device: &Self::Device, dtype: DType) -> bool {
        let client = R::client(device);

        let type_usage = client.properties().type_usage(dtype.into());
        // Same as `TypeUsage::all_scalar()`, but we make the usage explicit here
        type_usage.is_superset(
            TypeUsage::Buffer
                | TypeUsage::Conversion
                | TypeUsage::Arithmetic
                | TypeUsage::DotProduct,
        )
    }

    fn dtype_usage(device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        let client = R::client(device);

        let props = client.properties();
        let storage = dtype.into();
        let usage = props.type_usage(storage);

        let mut out = DTypeUsageSet::new();

        if usage.is_superset(TypeUsage::Buffer | TypeUsage::Conversion) {
            out |= DTypeUsage::Storage;
        }

        if usage.contains(TypeUsage::Arithmetic) {
            out |= DTypeUsage::Arithmetic;
        }

        let has_mma = |cfg: &MmaConfig| {
            cfg.a_type == storage || cfg.b_type == storage || cfg.cd_type == storage
        };
        if props.features.cmma.iter().any(has_mma) || props.features.mma.iter().any(has_mma) {
            out |= DTypeUsage::Accelerated;
        }

        out
    }
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> core::fmt::Debug
    for CubeBackend<R, F, I, BT>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CubeCLBackend")
    }
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Clone
    for CubeBackend<R, F, I, BT>
{
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Default
    for CubeBackend<R, F, I, BT>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<R: cubecl::Runtime> CubeRuntime for R
where
    R::Device: DeviceOps,
{
    type CubeDevice = R::Device;
    type CubeServer = R::Server;
}

#[cfg(not(feature = "fusion"))]
impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> BackendIr
    for CubeBackend<R, F, I, BT>
{
    type Handle = CubeTensor<R>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        handle.handle
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        handle.handle
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        handle.handle
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self> {
        handle.handle
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        tensor
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        tensor
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        tensor
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle {
        tensor
    }
}
