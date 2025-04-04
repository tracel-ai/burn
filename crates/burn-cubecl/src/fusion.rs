use crate::BoolElement;
use crate::element::CubeElement;
use crate::{CubeBackend, CubeRuntime, FloatElement, IntElement, kernel, tensor::CubeTensor};

use burn_cubecl_fusion::CubeFusionHandle;
use burn_cubecl_fusion::elemwise::optimization::ElemwiseOptimization;
use burn_cubecl_fusion::matmul::MatmulFallbackFn;
use burn_cubecl_fusion::matmul::builder::MatmulBuilder;
use burn_cubecl_fusion::matmul::optimization::MatmulOptimization;
use burn_cubecl_fusion::reduce::builder::ReduceBuilder;
use burn_cubecl_fusion::reduce::optimization::{
    ReduceFallbackFn, ReduceInstruction, ReduceOptimization,
};
use burn_cubecl_fusion::{
    CubeOptimization, CubeOptimizationState, elemwise::builder::ElementWiseBuilder,
};
use burn_fusion::{FusionBackend, FusionRuntime, client::MutexFusionClient};
use burn_ir::{BackendIr, TensorHandle};
use burn_tensor::{DType, Shape};
use core::marker::PhantomData;
use cubecl::reduce::instructions::ReduceFnConfig;
use half::{bf16, f16};
use std::sync::Arc;

impl<R, BT> burn_fusion::Optimization<FusionCubeRuntime<R, BT>> for CubeOptimization<R>
where
    R: CubeRuntime,
    BT: BoolElement,
{
    fn execute(&mut self, context: &mut burn_fusion::stream::Context<'_, CubeFusionHandle<R>>) {
        match self {
            Self::ElementWise(op) => op.execute::<BT>(context),
            Self::Matmul(op) => op.execute::<BT>(context),
            Self::Reduce(op) => op.execute::<BT>(context),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::ElementWise(op) => op.num_ops_fused(),
            Self::Matmul(op) => op.num_ops_fused(),
            Self::Reduce(op) => op.num_ops_fused(),
        }
    }

    fn to_state(&self) -> CubeOptimizationState {
        match self {
            Self::ElementWise(value) => CubeOptimizationState::ElementWise(value.to_state()),
            Self::Matmul(value) => CubeOptimizationState::Matmul(value.to_state()),
            Self::Reduce(value) => CubeOptimizationState::Reduce(value.to_state()),
        }
    }

    fn from_state(device: &R::Device, state: CubeOptimizationState) -> Self {
        match state {
            CubeOptimizationState::ElementWise(state) => {
                Self::ElementWise(ElemwiseOptimization::from_state(device, state))
            }
            CubeOptimizationState::Matmul(state) => Self::Matmul(MatmulOptimization::from_state(
                device,
                state,
                Arc::new(FallbackMatmul),
            )),
            CubeOptimizationState::Reduce(state) => Self::Reduce(ReduceOptimization::from_state(
                device,
                state,
                Arc::new(FallbackReduce),
            )),
        }
    }
}

struct FallbackMatmul;
struct FallbackReduce;

impl<R: CubeRuntime> MatmulFallbackFn<R> for FallbackMatmul {
    fn run(
        &self,
        lhs: (CubeFusionHandle<R>, &[usize]),
        rhs: (CubeFusionHandle<R>, &[usize]),
    ) -> CubeFusionHandle<R> {
        match lhs.0.dtype {
            DType::F64 => run_fallback_matmul::<R, f64>(lhs, rhs),
            DType::F32 => run_fallback_matmul::<R, f32>(lhs, rhs),
            DType::F16 => run_fallback_matmul::<R, f16>(lhs, rhs),
            DType::BF16 => run_fallback_matmul::<R, bf16>(lhs, rhs),
            _ => todo!("Not yet supported"),
        }
    }
}

impl<R: CubeRuntime> ReduceFallbackFn<R> for FallbackReduce {
    fn run(
        &self,
        input: CubeFusionHandle<R>,
        shape: &[usize],
        axis: usize,
        inst: &ReduceInstruction,
        d_o: &DType,
    ) -> CubeFusionHandle<R> {
        let d_i = input.dtype;
        let config = match inst {
            ReduceInstruction::ArgMax => ReduceFnConfig::ArgMax,
            ReduceInstruction::ArgMin => ReduceFnConfig::ArgMin,
            ReduceInstruction::Mean => ReduceFnConfig::Mean,
            ReduceInstruction::Prod => ReduceFnConfig::Prod,
            ReduceInstruction::Sum => ReduceFnConfig::Sum,
        };

        reduce_dtype::<R>(input, shape, axis, &d_i, d_o, config)
    }
}

fn run_fallback_matmul<R: CubeRuntime, EG: FloatElement>(
    lhs: (CubeFusionHandle<R>, &[usize]),
    rhs: (CubeFusionHandle<R>, &[usize]),
) -> CubeFusionHandle<R> {
    let lhs_tensor = into_tensor(
        lhs.0,
        Shape {
            dims: lhs.1.to_vec(),
        },
    );
    let rhs_tensor = into_tensor(
        rhs.0,
        Shape {
            dims: rhs.1.to_vec(),
        },
    );
    let out_tensor = crate::kernel::matmul::matmul::<R, EG>(
        lhs_tensor,
        rhs_tensor,
        None,
        crate::kernel::matmul::MatmulStrategy::default(),
    )
    .unwrap();

    CubeFusionHandle {
        client: out_tensor.client,
        handle: out_tensor.handle,
        device: out_tensor.device,
        dtype: out_tensor.dtype,
        strides: out_tensor.strides,
    }
}

fn reduce_dtype<R: CubeRuntime>(
    input_handle: CubeFusionHandle<R>,
    shape: &[usize],
    axis: usize,
    dtype_input: &DType,
    dtype_output: &DType,
    config: ReduceFnConfig,
) -> CubeFusionHandle<R> {
    match dtype_input {
        DType::F64 => {
            reduce_dtype_output::<R, f64>(input_handle, shape, axis, dtype_output, config)
        }
        DType::F32 => {
            reduce_dtype_output::<R, f32>(input_handle, shape, axis, dtype_output, config)
        }
        DType::F16 => {
            reduce_dtype_output::<R, f16>(input_handle, shape, axis, dtype_output, config)
        }
        DType::BF16 => {
            reduce_dtype_output::<R, bf16>(input_handle, shape, axis, dtype_output, config)
        }
        DType::I64 => {
            reduce_dtype_output::<R, i64>(input_handle, shape, axis, dtype_output, config)
        }
        DType::I32 => {
            reduce_dtype_output::<R, i32>(input_handle, shape, axis, dtype_output, config)
        }
        DType::I16 => {
            reduce_dtype_output::<R, i16>(input_handle, shape, axis, dtype_output, config)
        }
        DType::U64 => {
            reduce_dtype_output::<R, u64>(input_handle, shape, axis, dtype_output, config)
        }
        DType::U32 => {
            reduce_dtype_output::<R, u32>(input_handle, shape, axis, dtype_output, config)
        }
        DType::U16 => {
            reduce_dtype_output::<R, u16>(input_handle, shape, axis, dtype_output, config)
        }
        _ => todo!("Not yet supported"),
    }
}

fn reduce_dtype_output<R: CubeRuntime, In: CubeElement>(
    input_handle: CubeFusionHandle<R>,
    shape: &[usize],
    axis: usize,
    dtype_output: &DType,
    config: ReduceFnConfig,
) -> CubeFusionHandle<R> {
    match dtype_output {
        DType::F64 => reduce::<R, In, f64>(input_handle, shape, axis, config),
        DType::F32 => reduce::<R, In, f32>(input_handle, shape, axis, config),
        DType::F16 => reduce::<R, In, f16>(input_handle, shape, axis, config),
        DType::BF16 => reduce::<R, In, bf16>(input_handle, shape, axis, config),
        DType::I64 => reduce::<R, In, i64>(input_handle, shape, axis, config),
        DType::I32 => reduce::<R, In, i32>(input_handle, shape, axis, config),
        DType::I16 => reduce::<R, In, i16>(input_handle, shape, axis, config),
        DType::U64 => reduce::<R, In, u64>(input_handle, shape, axis, config),
        DType::U32 => reduce::<R, In, u32>(input_handle, shape, axis, config),
        DType::U16 => reduce::<R, In, u16>(input_handle, shape, axis, config),
        _ => todo!("Not yet supported"),
    }
}

fn reduce<R: CubeRuntime, In: CubeElement, Out: CubeElement>(
    input_handle: CubeFusionHandle<R>,
    shape: &[usize],
    axis: usize,
    config: ReduceFnConfig,
) -> CubeFusionHandle<R> {
    let input_tensor = into_tensor(
        input_handle,
        Shape {
            dims: shape.to_vec(),
        },
    );
    let out_tensor = crate::kernel::reduce::reduce_dim::<R, In, Out>(
        input_tensor,
        axis,
        crate::kernel::reduce::ReduceStrategy::default(),
        config,
    )
    .unwrap();

    CubeFusionHandle {
        client: out_tensor.client,
        handle: out_tensor.handle,
        device: out_tensor.device,
        dtype: out_tensor.dtype,
        strides: out_tensor.strides,
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
    type FusionClient = MutexFusionClient<Self>;
    type BoolRepr = BT;

    fn optimizations(
        device: R::Device,
    ) -> Vec<Box<dyn burn_fusion::OptimizationBuilder<Self::Optimization>>> {
        vec![
            Box::new(ElementWiseBuilder::<R>::new(
                device.clone(),
                BT::as_elem_native_unchecked().into(),
            )),
            Box::new(MatmulBuilder::<R>::new(
                device.clone(),
                BT::as_elem_native_unchecked().into(),
                Arc::new(FallbackMatmul),
            )),
            Box::new(ReduceBuilder::<R>::new(
                device.clone(),
                BT::as_elem_native_unchecked().into(),
                Arc::new(FallbackReduce),
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
        fn cast<R: CubeRuntime, F: FloatElement, FTarget: FloatElement>(
            tensor: CubeTensor<R>,
        ) -> CubeFusionHandle<R> {
            CubeFusionHandle::from(kernel::cast::<R, F, FTarget>(tensor))
        }

        match dtype {
            DType::F32 => cast::<R, F, f32>(tensor),
            DType::F16 => cast::<R, F, f16>(tensor),
            DType::BF16 => cast::<R, F, bf16>(tensor),
            _ => panic!("Casting error: {dtype:?} unsupported."),
        }
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
        }
    }
}
