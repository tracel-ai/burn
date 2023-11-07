use crate::{
    client::FusionClient, graph::TensorOpsDescription, FusionClientLocator, FusionTensor,
    HandleContainer,
};
use burn_tensor::{backend::Backend, Shape};
use core::marker::PhantomData;
use std::sync::Arc;

static CLIENTS: FusionClientLocator = FusionClientLocator::new();

#[derive(Clone, Debug, Default)]
pub struct Fusion<B> {
    _backend: PhantomData<B>,
}

impl<B: FusedBackend> Backend for Fusion<B> {
    type Device = B::Device;

    type FullPrecisionBackend = Fusion<B::FullPrecisionFusedBackend>;

    type FullPrecisionElem = B::FullPrecisionElem;

    type TensorPrimitive<const D: usize> = FusionTensor<B::FusionClient>;

    type FloatElem = B::FloatElem;

    type IntTensorPrimitive<const D: usize> = FusionTensor<B::FusionClient>;

    type IntElem = B::IntElem;

    type BoolTensorPrimitive<const D: usize> = FusionTensor<B::FusionClient>;

    fn name() -> String {
        todo!()
    }

    fn seed(seed: u64) {
        todo!()
    }
}

pub enum FusionStatus {
    /// No more operation can be fused.
    Closed(FusionProperties),
    /// More operations can be fused.
    Open(FusionProperties),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FusionProperties {
    pub score: u64,
    pub ready: bool,
}

pub trait FusedOps<B: FusedBackend>: Send {
    fn register(&mut self, ops: Arc<TensorOpsDescription<B>>) -> FusionStatus;
    fn execute(&mut self, handles: &mut HandleContainer<B>);
    fn reset(&mut self);
    fn len(&self) -> usize;
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, new)]
pub struct DeviceId {
    pub type_id: u16,
    pub index_id: u32,
}

pub trait HandleDevice: Clone + Send + Sync + PartialEq {
    fn id(&self) -> DeviceId;
}

pub trait FusedBackend: Backend {
    type HandleDevice: HandleDevice + From<Self::Device> + Into<Self::Device>;
    type Handle: Sync + Send + Clone;

    type FullPrecisionFusedBackend: FusedBackend<
            Handle = Self::Handle,
            Device = Self::Device,
            HandleDevice = Self::HandleDevice,
        > + Backend<Device = Self::Device, FloatElem = Self::FullPrecisionElem>;

    fn operations() -> Vec<Box<dyn FusedOps<Self>>>;
    fn float_tensor<const D: usize>(
        handle: Self::Handle,
        shape: Shape<D>,
    ) -> Self::TensorPrimitive<D>;
    fn int_tensor<const D: usize>(
        handle: Self::Handle,
        shape: Shape<D>,
    ) -> Self::IntTensorPrimitive<D>;
    fn bool_tensor<const D: usize>(
        handle: Self::Handle,
        shape: Shape<D>,
    ) -> Self::BoolTensorPrimitive<D>;

    fn float_tensor_handle<const D: usize>(tensor: Self::TensorPrimitive<D>) -> Self::Handle;
    fn int_tensor_handle<const D: usize>(tensor: Self::IntTensorPrimitive<D>) -> Self::Handle;
    fn bool_tensor_handle<const D: usize>(tensor: Self::BoolTensorPrimitive<D>) -> Self::Handle;

    type FusionClient: FusionClient<FusedBackend = Self>;

    fn client(device: &Self::HandleDevice) -> Self::FusionClient {
        CLIENTS.client(device)
    }
}
