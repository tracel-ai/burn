use alloc::sync::Arc;
use burn_common::stream::StreamId;
use spin::Mutex;

use crate::{
    repr::{
        HandleContainer, OperationDescription, ReprBackend, TensorDescription, TensorId,
        TensorStatus,
    },
    DType, TensorData,
};

use super::{RouterTensor, RunnerClient};

pub struct RunnerContext<B: ReprBackend> {
    handles: HandleContainer<B::Handle>,
}

impl<B: ReprBackend> RunnerContext<B> {
    pub fn create_empty_handle(&mut self) -> Arc<TensorId> {
        self.handles.create_tensor_uninit()
    }
}

// HttpChannel will probably need a different RunnerClient implementation (otherwise, the
// client machine will have backend requirements that could only be attributed to the server/runner)

// Runners are used by the channels (e.g., DirectChannel, HttpChannel, etc.)
// Responsible for executing the tensor operations.
#[derive(Clone)]
pub struct Runner<B: ReprBackend> {
    // Mutex for the mutable handles
    context: Arc<Mutex<RunnerContext<B>>>,
    device: B::Device,
}

impl<B: ReprBackend> Runner<B> {
    pub(crate) fn new(device: B::Device) -> Self {
        Self {
            context: Arc::new(Mutex::new(RunnerContext {
                handles: HandleContainer::new(),
            })),
            device,
        }
    }

    pub(crate) fn get_float_tensor(&self, tensor: &TensorDescription) -> B::Handle {
        let handles = &mut self.context.lock().handles;
        // handles.get_float_tensor::<B>(tensor)
        handles.get_tensor_handle(tensor).handle
    }

    pub(crate) fn get_int_tensor(&self, tensor: &TensorDescription) -> B::Handle {
        todo!()
    }

    pub(crate) fn get_bool_tensor(&self, tensor: &TensorDescription) -> B::Handle {
        todo!()
    }

    pub(crate) fn get_quantized_tensor(
        &self,
        tensor: TensorDescription,
    ) -> B::QuantizedTensorPrimitive {
        todo!()
    }

    /// Create a tensor with the given handle and shape.
    pub(crate) fn register_tensor(
        &self,
        handle: B::Handle,
        shape: Vec<usize>,
        dtype: DType,
    ) -> TensorDescription {
        let mut ctx = self.context.lock();
        let id = ctx.create_empty_handle();

        ctx.handles.register_handle(*id.as_ref(), handle);
        core::mem::drop(ctx);

        TensorDescription {
            id: *id.as_ref(),
            shape,
            status: TensorStatus::ReadOnly,
            dtype,
        }
    }
}

impl<B: ReprBackend> RunnerClient for Runner<B> {
    type Device = B::Device;

    /// Execute a tensor operation.
    fn register(&self, op: OperationDescription) {
        match op {
            OperationDescription::BaseFloat(_) => todo!(),
            OperationDescription::BaseInt(_) => todo!(),
            OperationDescription::BaseBool(_) => todo!(),
            OperationDescription::NumericFloat(_dtype, op) => match op {
                crate::repr::NumericOperationDescription::Add(desc) => {
                    let handles = &mut self.context.lock().handles;
                    // TODO: macro like fusion for `binary_float_ops!` etc.
                    let lhs = handles.get_float_tensor::<B>(&desc.lhs);
                    let rhs = handles.get_float_tensor::<B>(&desc.rhs);

                    let output = B::float_add(lhs, rhs);

                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                crate::repr::NumericOperationDescription::AddScalar(_) => todo!(),
                crate::repr::NumericOperationDescription::Sub(_) => todo!(),
                crate::repr::NumericOperationDescription::SubScalar(_) => todo!(),
                crate::repr::NumericOperationDescription::Div(_) => todo!(),
                crate::repr::NumericOperationDescription::DivScalar(_) => todo!(),
                crate::repr::NumericOperationDescription::RemScalar(_) => todo!(),
                crate::repr::NumericOperationDescription::Mul(_) => todo!(),
                crate::repr::NumericOperationDescription::MulScalar(_) => todo!(),
                crate::repr::NumericOperationDescription::Abs(_) => todo!(),
                crate::repr::NumericOperationDescription::Ones(_) => todo!(),
                crate::repr::NumericOperationDescription::Zeros(_) => todo!(),
                crate::repr::NumericOperationDescription::Full(_) => todo!(),
                crate::repr::NumericOperationDescription::Gather(_) => todo!(),
                crate::repr::NumericOperationDescription::Scatter(_) => todo!(),
                crate::repr::NumericOperationDescription::Select(_) => todo!(),
                crate::repr::NumericOperationDescription::SelectAssign(_) => todo!(),
                crate::repr::NumericOperationDescription::MaskWhere(_) => todo!(),
                crate::repr::NumericOperationDescription::MaskFill(_) => todo!(),
                crate::repr::NumericOperationDescription::MeanDim(_) => todo!(),
                crate::repr::NumericOperationDescription::Mean(_) => todo!(),
                crate::repr::NumericOperationDescription::Sum(_) => todo!(),
                crate::repr::NumericOperationDescription::SumDim(_) => todo!(),
                crate::repr::NumericOperationDescription::Prod(_) => todo!(),
                crate::repr::NumericOperationDescription::ProdDim(_) => todo!(),
                crate::repr::NumericOperationDescription::EqualElem(_) => todo!(),
                crate::repr::NumericOperationDescription::Greater(_) => todo!(),
                crate::repr::NumericOperationDescription::GreaterElem(_) => todo!(),
                crate::repr::NumericOperationDescription::GreaterEqual(_) => todo!(),
                crate::repr::NumericOperationDescription::GreaterEqualElem(_) => todo!(),
                crate::repr::NumericOperationDescription::Lower(_) => todo!(),
                crate::repr::NumericOperationDescription::LowerElem(_) => todo!(),
                crate::repr::NumericOperationDescription::LowerEqual(_) => todo!(),
                crate::repr::NumericOperationDescription::LowerEqualElem(_) => todo!(),
                crate::repr::NumericOperationDescription::ArgMax(_) => todo!(),
                crate::repr::NumericOperationDescription::ArgMin(_) => todo!(),
                crate::repr::NumericOperationDescription::Max(_) => todo!(),
                crate::repr::NumericOperationDescription::MaxDimWithIndices(_) => todo!(),
                crate::repr::NumericOperationDescription::MinDimWithIndices(_) => todo!(),
                crate::repr::NumericOperationDescription::Min(_) => todo!(),
                crate::repr::NumericOperationDescription::MaxDim(_) => todo!(),
                crate::repr::NumericOperationDescription::MinDim(_) => todo!(),
                crate::repr::NumericOperationDescription::Clamp(_) => todo!(),
                crate::repr::NumericOperationDescription::IntRandom(_) => todo!(),
                crate::repr::NumericOperationDescription::Powf(_) => todo!(),
            },
            OperationDescription::NumericInt(_, _) => todo!(),
            OperationDescription::Bool(_) => todo!(),
            OperationDescription::Int(_) => todo!(),
            OperationDescription::Float(_dtype, op) => match op {
                crate::repr::FloatOperationDescription::Exp(_) => todo!(),
                crate::repr::FloatOperationDescription::Log(_) => todo!(),
                crate::repr::FloatOperationDescription::Log1p(_) => todo!(),
                crate::repr::FloatOperationDescription::Erf(_) => todo!(),
                crate::repr::FloatOperationDescription::PowfScalar(_) => todo!(),
                crate::repr::FloatOperationDescription::Sqrt(_) => todo!(),
                crate::repr::FloatOperationDescription::Cos(_) => todo!(),
                crate::repr::FloatOperationDescription::Sin(_) => todo!(),
                crate::repr::FloatOperationDescription::Tanh(_) => todo!(),
                crate::repr::FloatOperationDescription::IntoInt(_) => todo!(),
                crate::repr::FloatOperationDescription::Matmul(_) => todo!(),
                crate::repr::FloatOperationDescription::Random(_desc) => {
                    todo!() // B::float_random(shape, distribution, device)
                }
                crate::repr::FloatOperationDescription::Recip(_) => todo!(),
                crate::repr::FloatOperationDescription::Quantize(_) => todo!(),
                crate::repr::FloatOperationDescription::Dequantize(_) => todo!(),
            },
            OperationDescription::Module(_) => todo!(),
        }
    }

    async fn read_tensor(&self, tensor: TensorDescription) -> TensorData {
        let mut ctx = self.context.lock();
        let tensor = ctx.handles.get_float_tensor::<B>(&tensor);

        B::float_into_data(tensor).await
    }

    fn write_tensor(&self, data: TensorData) -> TensorDescription {
        let mut ctx = self.context.lock();
        let id = ctx.create_empty_handle();
        let shape = data.shape.clone();
        let dtype = data.dtype;

        // TODO: split into write_float_tensor, write_int_tensor, ... ?
        match &data.dtype {
            DType::F64 | DType::F32 | DType::F16 | DType::BF16 => {
                // TODO: maybe device should be an arg to write_tensor?
                let tensor = B::float_from_data(data, &self.device);
                ctx.handles.register_float_tensor::<B>(&id, tensor)
            }
            _ => todo!(),
        }

        core::mem::drop(ctx);

        TensorDescription {
            id: *id,
            shape,
            status: TensorStatus::ReadOnly,
            dtype,
        }
    }

    fn register_new_tensor(&self, shape: Vec<usize>, dtype: DType) -> RouterTensor<Self> {
        let mut ctx = self.context.lock();
        let id = ctx.create_empty_handle();
        core::mem::drop(ctx);

        // TODO: register_new_tensor for output should initialize w/ handle
        // (i.e., do not use create_empty_handle which calls handles.create_tensor_uninit)

        // ctx.handles.register_handle(*id.as_ref(), handle);
        // core::mem::drop(ctx);

        let desc = TensorDescription {
            id: *id,
            shape: shape,
            status: TensorStatus::ReadOnly, // right status? uninit doesn't get overwritten in op
            dtype: dtype,
        };

        RouterTensor {
            desc,
            client: self.clone(),
        }
    }

    fn device(&self) -> Self::Device {
        self.device.clone()
    }
}
