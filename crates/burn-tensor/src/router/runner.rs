use alloc::sync::Arc;
use spin::Mutex;

use crate::{
    repr::{
        FloatOperationDescription, HandleContainer, NumericOperationDescription,
        OperationDescription, ReprBackend, TensorDescription, TensorId,
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
    pub(crate) fn register_tensor<C: RunnerClient>(
        &self,
        handle: B::Handle,
        shape: Vec<usize>,
        dtype: DType,
        client: C,
    ) -> RouterTensor<C> {
        let mut ctx = self.context.lock();
        let id = ctx.create_empty_handle();

        ctx.handles.register_handle(*id.as_ref(), handle);
        core::mem::drop(ctx);

        RouterTensor {
            id,
            shape,
            dtype,
            client,
        }
    }
}

impl<B: ReprBackend> RunnerClient for Runner<B> {
    type Device = B::Device;

    /// Execute a tensor operation.
    fn register(&self, op: OperationDescription) {
        match &op {
            OperationDescription::BaseFloat(_) => todo!(),
            OperationDescription::BaseInt(_) => todo!(),
            OperationDescription::BaseBool(_) => todo!(),
            OperationDescription::NumericFloat(_dtype, op) => match op {
                NumericOperationDescription::Add(desc) => {
                    let handles = &mut self.context.lock().handles;
                    // TODO: macro like fusion for `binary_float_ops!` etc.
                    let lhs = handles.get_float_tensor::<B>(&desc.lhs);
                    let rhs = handles.get_float_tensor::<B>(&desc.rhs);

                    // In fusion: out = client.tensor_uninitialized(...) yields Handle::NotInit
                    // output: FusionTensor
                    let output = B::float_add(lhs, rhs);

                    // B::sync(
                    //     &B::float_device(&output),
                    //     burn_common::sync_type::SyncType::Wait,
                    // ); // force fusion output tensor handle registration (just to confirm that it works!)

                    // B::float_tensor_handle(output) -> Fusion::float_tensor_handle(tensor) ->
                    // trying to get a handle from container that is not initialized (NotInit)
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationDescription::AddScalar(_) => todo!(),
                NumericOperationDescription::Sub(_) => todo!(),
                NumericOperationDescription::SubScalar(_) => todo!(),
                NumericOperationDescription::Div(_) => todo!(),
                NumericOperationDescription::DivScalar(_) => todo!(),
                NumericOperationDescription::RemScalar(_) => todo!(),
                NumericOperationDescription::Mul(_) => todo!(),
                NumericOperationDescription::MulScalar(_) => todo!(),
                NumericOperationDescription::Abs(_) => todo!(),
                NumericOperationDescription::Ones(_) => todo!(),
                NumericOperationDescription::Zeros(_) => todo!(),
                NumericOperationDescription::Full(_) => todo!(),
                NumericOperationDescription::Gather(_) => todo!(),
                NumericOperationDescription::Scatter(_) => todo!(),
                NumericOperationDescription::Select(_) => todo!(),
                NumericOperationDescription::SelectAssign(_) => todo!(),
                NumericOperationDescription::MaskWhere(_) => todo!(),
                NumericOperationDescription::MaskFill(_) => todo!(),
                NumericOperationDescription::MeanDim(_) => todo!(),
                NumericOperationDescription::Mean(_) => todo!(),
                NumericOperationDescription::Sum(_) => todo!(),
                NumericOperationDescription::SumDim(_) => todo!(),
                NumericOperationDescription::Prod(_) => todo!(),
                NumericOperationDescription::ProdDim(_) => todo!(),
                NumericOperationDescription::EqualElem(_) => todo!(),
                NumericOperationDescription::Greater(_) => todo!(),
                NumericOperationDescription::GreaterElem(_) => todo!(),
                NumericOperationDescription::GreaterEqual(_) => todo!(),
                NumericOperationDescription::GreaterEqualElem(_) => todo!(),
                NumericOperationDescription::Lower(_) => todo!(),
                NumericOperationDescription::LowerElem(_) => todo!(),
                NumericOperationDescription::LowerEqual(_) => todo!(),
                NumericOperationDescription::LowerEqualElem(_) => todo!(),
                NumericOperationDescription::ArgMax(_) => todo!(),
                NumericOperationDescription::ArgMin(_) => todo!(),
                NumericOperationDescription::Max(_) => todo!(),
                NumericOperationDescription::MaxDimWithIndices(_) => todo!(),
                NumericOperationDescription::MinDimWithIndices(_) => todo!(),
                NumericOperationDescription::Min(_) => todo!(),
                NumericOperationDescription::MaxDim(_) => todo!(),
                NumericOperationDescription::MinDim(_) => todo!(),
                NumericOperationDescription::Clamp(_) => todo!(),
                NumericOperationDescription::IntRandom(_) => todo!(),
                NumericOperationDescription::Powf(_) => todo!(),
            },
            OperationDescription::NumericInt(_, _) => todo!(),
            OperationDescription::Bool(_) => todo!(),
            OperationDescription::Int(_) => todo!(),
            OperationDescription::Float(_dtype, op) => match op {
                FloatOperationDescription::Exp(_) => todo!(),
                FloatOperationDescription::Log(_) => todo!(),
                FloatOperationDescription::Log1p(_) => todo!(),
                FloatOperationDescription::Erf(_) => todo!(),
                FloatOperationDescription::PowfScalar(_) => todo!(),
                FloatOperationDescription::Sqrt(_) => todo!(),
                FloatOperationDescription::Cos(_) => todo!(),
                FloatOperationDescription::Sin(_) => todo!(),
                FloatOperationDescription::Tanh(_) => todo!(),
                FloatOperationDescription::IntoInt(_) => todo!(),
                FloatOperationDescription::Matmul(_) => todo!(),
                FloatOperationDescription::Random(_desc) => {
                    todo!() // B::float_random(shape, distribution, device)
                }
                FloatOperationDescription::Recip(_) => todo!(),
                FloatOperationDescription::Quantize(_) => todo!(),
                FloatOperationDescription::Dequantize(_) => todo!(),
            },
            OperationDescription::Module(_) => todo!(),
        }

        // Remove unused tensor handles
        // NOTE: only ReadWrite handles are removed
        let mut ctx = self.context.lock();
        op.nodes()
            .into_iter()
            .for_each(|tensor| ctx.handles.free(tensor));
    }

    async fn read_tensor(&self, tensor: TensorDescription) -> TensorData {
        let mut ctx = self.context.lock();
        let tensor = ctx.handles.get_float_tensor::<B>(&tensor);

        B::float_into_data(tensor).await
    }

    fn write_tensor(&self, data: TensorData) -> RouterTensor<Self> {
        let mut ctx = self.context.lock();
        let id = ctx.create_empty_handle();
        let shape = data.shape.clone();
        let dtype = data.dtype;

        // TODO: split into write_float_tensor, write_int_tensor, ... ?
        match dtype {
            DType::F64 | DType::F32 | DType::F16 | DType::BF16 => {
                let tensor = B::float_from_data(data, &self.device);
                ctx.handles.register_float_tensor::<B>(&id, tensor)
            }
            _ => todo!(),
        }

        core::mem::drop(ctx);

        RouterTensor {
            id,
            shape,
            dtype,
            client: self.clone(),
        }
    }

    fn register_new_tensor(&self, shape: Vec<usize>, dtype: DType) -> RouterTensor<Self> {
        let mut ctx = self.context.lock();
        let id = ctx.create_empty_handle();
        core::mem::drop(ctx);

        // TODO: register_new_tensor for output should initialize w/ handle
        // (i.e., do not use create_empty_handle which calls handles.create_tensor_uninit)
        // NOTE: not true. sure, that's true for the current implementation (local backend runner),
        // but in general the operations could be async - so the handle only exists when the output
        // has been computed.

        RouterTensor {
            id,
            shape,
            dtype,
            client: self.clone(),
        }
    }

    fn device(&self) -> Self::Device {
        self.device.clone()
    }
}
