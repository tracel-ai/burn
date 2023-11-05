use super::TensorOps;
use crate::{client::FusionClient, Fusion, HandleContainer};
use burn_tensor::backend::Backend;
use std::{ops::RangeBounds, sync::Arc, vec::Drain};

pub struct Graph<B: FusedBackend> {
    operations: Vec<Arc<TensorOps<B>>>,
}

impl<B: FusedBackend> Graph<B> {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }
    pub fn add(&mut self, ops: Arc<TensorOps<B>>) {
        self.operations.push(ops);
    }

    pub fn len(&self) -> usize {
        self.operations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.operations.len() == 0
    }

    fn drain<R>(&mut self, range: R) -> Drain<'_, Arc<TensorOps<B>>>
    where
        R: RangeBounds<usize>,
    {
        self.operations.drain(range)
    }

    pub fn remove<R: RangeBounds<usize>>(&mut self, range: R, handles: &mut HandleContainer<B>) {
        for ops in self.operations.drain(range) {
            ops.cleanup_tensor(handles)
        }
    }

    pub fn nodes<'a>(&'a self) -> &'a [Arc<TensorOps<B>>] {
        &self.operations
    }

    pub fn execute_optimization(
        &mut self,
        handles: &mut HandleContainer<B>,
        index: usize,
        optimizations: &mut [Optimization<B>],
    ) {
        let optimization = optimizations.get_mut(index).unwrap();
        let num_keep = optimization.ops.len();
        optimization.ops.execute(handles);

        self.remove(0..num_keep, handles);

        for optimization in optimizations.iter_mut() {
            optimization.reset();

            for node in self.nodes() {
                optimization.register(node);
            }
        }
    }

    pub fn execute(&mut self, handles: &mut HandleContainer<B>) {
        for ops in self.drain(..) {
            todo!("execute the ops");
            ops.cleanup_tensor(handles);
        }
    }
}

#[derive(new)]
pub struct Optimization<B: FusedBackend> {
    pub ops: Box<dyn FusedOps<B>>,
    pub status: FusionStatus,
}

impl<B: FusedBackend> Optimization<B> {
    pub fn register(&mut self, ops: &Arc<TensorOps<B>>) {
        match self.status {
            FusionStatus::Closed(_) => return,
            _ => {}
        };

        self.status = self.ops.register(ops.clone());
    }

    pub fn reset(&mut self) {
        self.ops.reset();
        self.status = FusionStatus::Open(FusionProperties::default());
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
    const FUSION: Fusion<Self::FusionClient> = Fusion::new();
}
