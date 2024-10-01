use alloc::sync::Arc;
use burn_common::stream::StreamId;
use spin::Mutex;

use crate::{
    repr::{HandleContainer, OperationDescription, ReprBackend, TensorDescription},
    DType, TensorData,
};

use super::RunnerClient;

pub struct RunnerContext<B: ReprBackend> {
    handles: HandleContainer<B::Handle>,
    device: B::Device,
}

// Runners are used by the channels (e.g., DirectChannel, HttpChannel, etc.)
// Responsible for executing the tensor operations.
#[derive(Clone)]
pub struct Runner<B: ReprBackend> {
    // Mutex for the mutable handles
    context: Arc<Mutex<RunnerContext<B>>>,
}

impl<B: ReprBackend> Runner<B> {
    pub(crate) fn new(device: B::Device) -> Self {
        Self {
            context: Arc::new(Mutex::new(RunnerContext {
                handles: HandleContainer::new(),
                device,
            })),
        }
    }
}

impl<B: ReprBackend> RunnerClient for Runner<B> {
    /// Execute a tensor operation.
    fn register(&self, op: OperationDescription, stream: StreamId) {
        match op {
            OperationDescription::BaseFloat(_) => todo!(),
            OperationDescription::BaseInt(_) => todo!(),
            OperationDescription::BaseBool(_) => todo!(),
            OperationDescription::NumericFloat(_, _) => todo!(),
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

    async fn read_tensor(&self, tensor: TensorDescription, stream: StreamId) -> TensorData {
        todo!()
    }

    fn write_tensor(&self, data: TensorData, stream: StreamId) -> TensorDescription {
        todo!()
    }

    fn empty_tensor(&self, shape: Vec<usize>, dtype: DType, stream: StreamId) -> TensorDescription {
        todo!()
    }
}
