use crate::{
    client::FusionClient, graph::TensorOps, FusionClientLocator, FusionTensor, HandleContainer,
};
use burn_tensor::backend::Backend;
use core::marker::PhantomData;
use std::sync::Arc;

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
    fn register(&mut self, ops: Arc<TensorOps<B>>) -> FusionStatus;
    fn execute(&mut self, handles: &mut HandleContainer<B>);
    fn reset(&mut self);
    fn len(&self) -> usize;
}

pub trait FusedBackend: Backend {
    type HandleDevice: core::hash::Hash
        + core::cmp::Eq
        + Clone
        + Send
        + Sync
        + core::fmt::Debug
        + From<Self::Device>
        + Into<Self::Device>;
    type Handle: Sync + Send + Clone;

    type FullPrecisionFusedBackend: FusedBackend<
            Handle = Self::Handle,
            Device = Self::Device,
            HandleDevice = Self::HandleDevice,
        > + Backend<Device = Self::Device, FloatElem = Self::FullPrecisionElem>;

    fn operations() -> Vec<Box<dyn FusedOps<Self>>>;
    fn new(shape: Vec<usize>) -> Self::Handle;

    fn float_tensor<const D: usize>(handle: Self::Handle) -> Self::TensorPrimitive<D>;
    fn int_tensor<const D: usize>(handle: Self::Handle) -> Self::IntTensorPrimitive<D>;
    fn bool_tensor<const D: usize>(handle: Self::Handle) -> Self::BoolTensorPrimitive<D>;

    fn float_tensor_handle<const D: usize>(tensor: Self::TensorPrimitive<D>) -> Self::Handle;
    fn int_tensor_handle<const D: usize>(tensor: Self::IntTensorPrimitive<D>) -> Self::Handle;
    fn bool_tensor_handle<const D: usize>(tensor: Self::BoolTensorPrimitive<D>) -> Self::Handle;

    type FusionClient: FusionClient<FusedBackend = Self>;
    const FUSION: FusionClientLocator<Self::FusionClient> = FusionClientLocator::new();
}
