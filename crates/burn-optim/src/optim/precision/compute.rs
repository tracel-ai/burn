use burn_core::tensor::{DType, FloatDType, Tensor};

/// Select the operand precision for an update and return the weight in the model dtype.
#[derive(Clone, Copy)]
pub(crate) enum ComputePrecision {
    Input,
    PromoteHalf,
    Fixed(FloatDType),
}

impl ComputePrecision {
    pub(crate) fn apply<const D: usize, S>(
        self,
        model: Tensor<D>,
        grad: Tensor<D>,
        update: impl FnOnce(Tensor<D>, Tensor<D>) -> (Tensor<D>, S),
    ) -> (Tensor<D>, S) {
        let dtype = match self {
            Self::Fixed(dtype) => dtype,
            Self::PromoteHalf if matches!(model.dtype(), DType::F16 | DType::BF16) => {
                FloatDType::F32
            }
            Self::Input | Self::PromoteHalf => return update(model, grad),
        };
        let model_dtype = model.dtype();
        let (updated, state) = update(model.cast(dtype), grad.cast(dtype));
        (updated.cast(model_dtype), state)
    }
}
