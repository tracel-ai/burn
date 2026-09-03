use crate::{CubeBackend, CubeDevice, kernel, tensor::CubeTensor};
use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use burn_backend::{DType, Shape};
pub use burn_cubecl_fusion::{CubeFusionHandle, FallbackOperation};
use burn_fusion::{
    FusionBackend, FusionRuntime,
    stream::{FallbackOp, OrderedExecution},
};
use burn_ir::{BackendIr, TensorHandle};
use burn_std::Metadata;
use core::marker::PhantomData;

mod registry;
pub use burn_cubecl_fusion::optim::{CubeOptimization, CubeOptimizationState, FusedOperation};
pub use registry::{
    BUILTIN_NAMES, CubeFuser, OptimizationProvider, RegistryError, register, remove,
};

impl burn_fusion::Optimization<FusionCubeRuntime> for CubeOptimization {
    fn execute(
        &mut self,
        context: &mut burn_fusion::stream::Context<
            <FusionCubeRuntime as FusionRuntime>::FusionHandle,
        >,
        execution: &OrderedExecution<FusionCubeRuntime>,
    ) {
        self.run(context, &|index| {
            let operation = execution.operation_within_optimization(index);
            Box::new(FallbackOperationWrapper::new(operation))
        })
    }

    fn to_state(&self) -> CubeOptimizationState {
        Self::to_state(self)
    }

    fn from_state(device: &CubeDevice, state: CubeOptimizationState) -> Self {
        registry::restore(device, state)
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

impl FallbackOperation for FallbackOperationWrapper<FallbackOp<FusionCubeRuntime>> {
    fn run(&self, context: &mut burn_fusion::stream::Context<CubeFusionHandle>) {
        // Through `FallbackOp`, so unfused work inside a fused block obeys the
        // same rule as unfused work outside one: an operation whose input a
        // failure claims does not run, and its outputs take that failure.
        self.operation.execute(&mut context.handles);
    }
}

impl BackendIr for CubeBackend {
    type Handle = CubeFusionHandle;

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

impl FusionRuntime for FusionCubeRuntime {
    type OptimizationState = CubeOptimizationState;
    type Optimization = CubeOptimization;
    type FusionHandle = CubeFusionHandle;
    type FusionDevice = CubeDevice;

    fn fusers(device: CubeDevice) -> Vec<Box<dyn burn_fusion::OperationFuser<Self::Optimization>>> {
        registry::fusers(&device)
    }
}

/// Fusion runtime for JIT runtimes.
#[derive(Debug)]
pub struct FusionCubeRuntime;

impl FusionBackend for CubeBackend {
    type FusionRuntime = FusionCubeRuntime;

    type FullPrecisionBackend = CubeBackend;

    fn cast_float(tensor: FloatTensor<Self>, dtype: DType) -> Self::Handle {
        kernel::cast(tensor, dtype).into()
    }

    fn memory_persistent(device: &Self::Device, enabled: bool) {
        use cubecl::MemoryAllocationMode;

        let client = device.client();
        let mode = match enabled {
            true => MemoryAllocationMode::Persistent,
            false => MemoryAllocationMode::Auto,
        };
        // Safety: called from the fusion execution thread, whose stream is the
        // one every fused operation allocates on.
        unsafe { client.allocation_mode(mode) };
    }
}

fn into_tensor(handle: CubeFusionHandle, shape: Shape) -> CubeTensor {
    CubeTensor {
        client: handle.client.clone(),
        handle: handle.handle.clone(),
        device: handle.device.clone(),
        meta: Box::new(Metadata::new(shape, handle.strides.clone())),
        dtype: handle.dtype,
        qparams: handle.qparams.clone(),
    }
}

impl From<CubeTensor> for CubeFusionHandle {
    fn from(value: CubeTensor) -> Self {
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
