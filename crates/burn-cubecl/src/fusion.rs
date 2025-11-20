use crate::BoolElement;
use crate::{CubeBackend, CubeRuntime, FloatElement, IntElement, kernel, tensor::CubeTensor};

use burn_cubecl_fusion::elemwise::optimization::ElemwiseOptimization;
use burn_cubecl_fusion::matmul::builder::MatmulBuilder;
use burn_cubecl_fusion::matmul::optimization::MatmulOptimization;
use burn_cubecl_fusion::reduce::builder::ReduceBuilder;
use burn_cubecl_fusion::reduce::optimization::ReduceOptimization;
use burn_cubecl_fusion::{CubeFusionHandle, FallbackOperation};
use burn_cubecl_fusion::{
    CubeOptimization, CubeOptimizationState, elemwise::builder::ElementWiseBuilder,
};
use burn_fusion::stream::{Operation, OrderedExecution};
use burn_fusion::{FusionBackend, FusionRuntime};
use burn_ir::{BackendIr, TensorHandle};
use burn_tensor::{DType, Shape};
use core::marker::PhantomData;
use std::sync::Arc;

impl<R, BT> burn_fusion::Optimization<FusionCubeRuntime<R, BT>> for CubeOptimization<R>
where
    R: CubeRuntime,
    BT: BoolElement,
{
    fn execute(
        &mut self,
        context: &mut burn_fusion::stream::Context<
            '_,
            <FusionCubeRuntime<R, BT> as FusionRuntime>::FusionHandle,
        >,
        execution: &OrderedExecution<FusionCubeRuntime<R, BT>>,
    ) {
        match self {
            Self::ElementWise(op) => op.execute::<BT>(context),
            Self::Matmul(op) => op.execute::<BT>(context, |index| {
                let operation = execution.operation_within_optimization(index);
                Box::new(FallbackOperationWrapper::new(operation))
            }),
            Self::Reduce(op) => op.execute::<BT>(context, |index| {
                let operation = execution.operation_within_optimization(index);
                Box::new(FallbackOperationWrapper::new(operation))
            }),
        }
    }

    fn to_state(&self) -> CubeOptimizationState {
        self.to_opt_state()
    }

    fn from_state(device: &R::Device, state: CubeOptimizationState) -> Self {
        match state {
            CubeOptimizationState::ElementWise(state) => {
                Self::ElementWise(ElemwiseOptimization::from_state(device, state))
            }
            CubeOptimizationState::Matmul(state) => {
                Self::Matmul(MatmulOptimization::from_state(device, state))
            }
            CubeOptimizationState::Reduce(state) => {
                Self::Reduce(ReduceOptimization::from_state(device, state))
            }
        }
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

impl<R: CubeRuntime, BT: BoolElement> FallbackOperation<R>
    for FallbackOperationWrapper<Arc<dyn Operation<FusionCubeRuntime<R, BT>>>>
{
    fn run(&self, context: &mut burn_fusion::stream::Context<'_, CubeFusionHandle<R>>) {
        self.operation.as_ref().execute(context.handles);
    }
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> BackendIr
    for CubeBackend<R, F, I, BT>
{
    type Handle = CubeFusionHandle<R>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> burn_tensor::ops::FloatTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> burn_tensor::ops::IntTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> burn_tensor::ops::BoolTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn quantized_tensor(
        handle: TensorHandle<Self::Handle>,
    ) -> burn_tensor::ops::QuantizedTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn float_tensor_handle(tensor: burn_tensor::ops::FloatTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn int_tensor_handle(tensor: burn_tensor::ops::IntTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn bool_tensor_handle(tensor: burn_tensor::ops::BoolTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn quantized_tensor_handle(tensor: burn_tensor::ops::QuantizedTensor<Self>) -> Self::Handle {
        tensor.into()
    }
}

impl<R: CubeRuntime, BT: BoolElement> FusionRuntime for FusionCubeRuntime<R, BT> {
    type OptimizationState = CubeOptimizationState;
    type Optimization = CubeOptimization<R>;
    type FusionHandle = CubeFusionHandle<R>;
    type FusionDevice = R::CubeDevice;
    type BoolRepr = BT;

    fn optimizations(
        device: R::Device,
    ) -> Vec<Box<dyn burn_fusion::OptimizationBuilder<Self::Optimization>>> {
        vec![
            Box::new(ElementWiseBuilder::<R>::new(
                device.clone(),
                BT::as_type_native_unchecked().into(),
            )),
            Box::new(MatmulBuilder::<R>::new(
                device.clone(),
                BT::as_type_native_unchecked().into(),
            )),
            Box::new(ReduceBuilder::<R>::new(
                device.clone(),
                BT::as_type_native_unchecked().into(),
            )),
        ]
    }
}

/// Fusion runtime for JIT runtimes.
#[derive(Debug)]
pub struct FusionCubeRuntime<R: CubeRuntime, BT: BoolElement> {
    _b: PhantomData<R>,
    _bool: PhantomData<BT>,
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> FusionBackend
    for CubeBackend<R, F, I, BT>
{
    type FusionRuntime = FusionCubeRuntime<R, BT>;

    type FullPrecisionBackend = CubeBackend<R, f32, i32, BT>;

    fn cast_float(tensor: burn_tensor::ops::FloatTensor<Self>, dtype: DType) -> Self::Handle {
        kernel::cast::<R>(tensor, dtype).into()
    }
}

fn into_tensor<R: CubeRuntime>(handle: CubeFusionHandle<R>, shape: Shape) -> CubeTensor<R> {
    CubeTensor {
        client: handle.client,
        handle: handle.handle,
        device: handle.device,
        shape,
        strides: handle.strides,
        dtype: handle.dtype,
        qparams: handle.qparams,
    }
}

impl<R: CubeRuntime> From<CubeTensor<R>> for CubeFusionHandle<R> {
    fn from(value: CubeTensor<R>) -> Self {
        Self {
            client: value.client,
            handle: value.handle,
            device: value.device,
            strides: value.strides,
            dtype: value.dtype,
            qparams: value.qparams,
        }
    }
}
