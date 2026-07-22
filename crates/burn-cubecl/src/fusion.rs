use crate::{CubeBackend, CubeRuntime, kernel, tensor::CubeTensor};
use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use burn_backend::{DType, Shape};
use burn_cubecl_fusion::{
    CubeFusionHandle, FallbackOperation,
    optim::{CubeOptimization, CubeOptimizationState},
};
use burn_fusion::UnfusedOp;
use burn_fusion::{
    FusionBackend, FusionRuntime,
    stream::{Operation, OrderedExecution},
};
use burn_ir::{BackendIr, TensorHandle};
use burn_std::Metadata;
use core::marker::PhantomData;
use std::sync::Arc;

mod registry;
pub use registry::{BUILTIN_NAMES, OptimizationProvider, RegistryError, register, remove};

impl<R> burn_fusion::Optimization<FusionCubeRuntime<R>> for Box<dyn CubeOptimization<R>>
where
    R: CubeRuntime,
{
    fn execute(
        &mut self,
        context: &mut burn_fusion::stream::Context<
            <FusionCubeRuntime<R> as FusionRuntime>::FusionHandle,
        >,
        execution: &OrderedExecution<FusionCubeRuntime<R>>,
    ) {
        self.as_mut().execute(context, &|index| {
            let operation = execution.operation_within_optimization(index);
            Box::new(FallbackOperationWrapper::new(operation))
        })
    }

    fn to_state(&self) -> CubeOptimizationState {
        CubeOptimizationState {
            name: self.as_ref().name().to_string(),
        }
    }

    fn from_state(_device: &R::Device, state: CubeOptimizationState) -> Self {
        // Optimizations are rebuilt by their fusers; no stored plan is ever
        // deserialized for the cubecl fusion runtime.
        panic!(
            "fusion optimization `{}` cannot be restored from a serialized execution plan",
            state.name
        )
    }
}

struct FallbackOperationWrapper<O: Clone> {
    operation: O,
}

impl<O: Clone> FallbackOperationWrapper<O> {
    fn new(op: O) -> Self {
        Self { operation: op }
    }
}

impl<R: CubeRuntime> FallbackOperation<R>
    for FallbackOperationWrapper<Arc<dyn Operation<FusionCubeRuntime<R>>>>
{
    fn run(&self, context: &mut burn_fusion::stream::Context<CubeFusionHandle<R>>) {
        self.operation.as_ref().execute(&mut context.handles);
    }
}

impl<R: CubeRuntime> FallbackOperation<R>
    for FallbackOperationWrapper<UnfusedOp<FusionCubeRuntime<R>>>
{
    fn run(&self, context: &mut burn_fusion::stream::Context<CubeFusionHandle<R>>) {
        self.operation.execute(&mut context.handles);
    }
}

impl<R: CubeRuntime> BackendIr for CubeBackend<R> {
    type Handle = CubeFusionHandle<R>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle {
        tensor.into()
    }
}

impl<R: CubeRuntime> FusionRuntime for FusionCubeRuntime<R> {
    type OptimizationState = CubeOptimizationState;
    type Optimization = Box<dyn CubeOptimization<R>>;
    type FusionHandle = CubeFusionHandle<R>;
    type FusionDevice = R::CubeDevice;

    fn fusers(device: R::Device) -> Vec<Box<dyn burn_fusion::OperationFuser<Self::Optimization>>> {
        registry::fusers::<R>(&device)
    }
}

/// Fusion runtime for JIT runtimes.
#[derive(Debug)]
pub struct FusionCubeRuntime<R: CubeRuntime> {
    _b: PhantomData<R>,
}

impl<R: CubeRuntime> FusionBackend for CubeBackend<R> {
    type FusionRuntime = FusionCubeRuntime<R>;

    type FullPrecisionBackend = CubeBackend<R>;

    fn cast_float(tensor: FloatTensor<Self>, dtype: DType) -> Self::Handle {
        kernel::cast(tensor, dtype).into()
    }

    fn memory_persistent(device: &Self::Device, enabled: bool) {
        use cubecl::MemoryAllocationMode;

        let client = R::client(device);
        let mode = match enabled {
            true => MemoryAllocationMode::Persistent,
            false => MemoryAllocationMode::Auto,
        };
        // Safety: called from the fusion execution thread, whose stream is the
        // one every fused operation allocates on.
        unsafe { client.allocation_mode(mode) };
    }
}

fn into_tensor<R: CubeRuntime>(handle: CubeFusionHandle<R>, shape: Shape) -> CubeTensor<R> {
    CubeTensor {
        client: handle.client.clone(),
        handle: handle.handle.clone(),
        device: handle.device.clone(),
        meta: Box::new(Metadata::new(shape, handle.strides.clone())),
        dtype: handle.dtype,
        qparams: handle.qparams.clone(),
    }
}

impl<R: CubeRuntime> From<CubeTensor<R>> for CubeFusionHandle<R> {
    fn from(value: CubeTensor<R>) -> Self {
        Self {
            client: value.client.clone(),
            handle: value.handle.clone(),
            device: value.device.clone(),
            strides: value.meta.strides.clone(),
            dtype: value.dtype,
            qparams: value.qparams.clone(),
        }
    }
}
