use burn_backend::{
    TensorMetadata, TensorPrimitive,
    ops::{FloatTensorOps, QTensorOps},
};
use burn_dispatch::{Dispatch, DispatchTensor};
use burn_std::{DType, DTypeResolution, FloatDType, TensorType};

use super::{BridgeKind, BridgeTensor};

/// The type an operation's output takes, resolved from its float inputs, with the conversions
/// that bring the operands to the precision the operation computes in and the output to that
/// type.
///
/// Every float operation of the bridge goes through one: a combining operation resolves its
/// output from all its inputs ([`combine`](Self::combine)), an operation writing into a tensor
/// keeps that tensor's type ([`into`](Self::into)). Backends never see a propagation: they receive
/// operands already converted and return an output this brings to the resolved type.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Resolved {
    out: TensorType,
}

impl Resolved {
    /// The output of a combining operation — elementwise, a matmul, a concatenation, a reduction —
    /// over `inputs`, under the process's [`DTypeResolution`].
    ///
    /// # Panics
    ///
    /// Under [`DTypeResolution::Strict`], when two propagating inputs hold different dtypes.
    pub(crate) fn combine<'a>(inputs: impl IntoIterator<Item = &'a BridgeTensor>) -> Self {
        let inputs = inputs.into_iter().map(BridgeTensor::tensor_type);
        let out = TensorType::combine(inputs, DTypeResolution::current())
            .unwrap_or_else(|conflict| panic!("{conflict}"));
        Self { out }
    }

    /// The output of an operation writing into `target`: the target's own type.
    pub(crate) fn into(target: &BridgeTensor) -> Self {
        Self {
            out: target.tensor_type(),
        }
    }

    /// The float precision the operation computes in: the output's when it is float, `f32` when
    /// it is quantized.
    pub(crate) fn compute(&self) -> FloatDType {
        match self.out.dtype {
            DType::QFloat(_) => FloatDType::F32,
            dtype => dtype.into(),
        }
    }

    /// Whether every one of `operands` is quantized, so a backend's quantized operation can take
    /// them as they are stored.
    pub(crate) fn all_quantized<'a>(operands: impl IntoIterator<Item = &'a BridgeTensor>) -> bool {
        operands.into_iter().all(BridgeTensor::is_qfloat)
    }

    /// `operand` as the operation reads it: float, in [`compute`](Self::compute).
    pub(crate) fn operand(&self, operand: BridgeTensor) -> DispatchTensor {
        let compute = self.compute();
        let (kind, operand) = operand.into_parts();
        match kind {
            BridgeKind::Float if operand.dtype() == compute.into() => operand,
            BridgeKind::Float => Dispatch::float_cast(operand, compute),
            BridgeKind::QFloat => Dispatch::dequantize(operand, compute),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    /// A float output, brought to the resolved type.
    pub(crate) fn output(&self, out: DispatchTensor) -> BridgeTensor {
        self.primitive(TensorPrimitive::Float(out))
    }

    /// An output a backend returned float or quantized, brought to the resolved type: quantized
    /// when the type is, cast when its float precision differs. A quantized output already in
    /// the resolved format keeps its orientation.
    pub(crate) fn primitive(&self, out: TensorPrimitive<Dispatch>) -> BridgeTensor {
        let out = match (out, self.out.dtype) {
            (TensorPrimitive::QFloat(out), _) if self.out.same_format(out.dtype()) => {
                BridgeTensor::qfloat(out)
            }
            (TensorPrimitive::Float(out), DType::QFloat(scheme)) => {
                BridgeTensor::qfloat(Dispatch::quantize_dynamic(out, &scheme))
            }
            (TensorPrimitive::QFloat(out), DType::QFloat(scheme)) => {
                let out = Dispatch::dequantize(out, FloatDType::F32);
                BridgeTensor::qfloat(Dispatch::quantize_dynamic(out, &scheme))
            }
            (TensorPrimitive::QFloat(out), dtype) => {
                BridgeTensor::float(Dispatch::dequantize(out, dtype.into()))
            }
            (TensorPrimitive::Float(out), dtype) if out.dtype() == dtype => {
                BridgeTensor::float(out)
            }
            (TensorPrimitive::Float(out), dtype) => {
                BridgeTensor::float(Dispatch::float_cast(out, dtype.into()))
            }
        };
        out.with_propagation(self.out.propagation)
    }
}
