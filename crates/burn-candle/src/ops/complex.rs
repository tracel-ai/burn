use burn_backend::{ComplexTensor, TypedDevice, UnimplementedTensorPrimitive};

use crate::{Candle, CandleDevice, FloatCandleElement, IntCandleElement};

impl TypedDevice<Self> for Candle {
    fn complex_device(tensor: &ComplexTensor<Self>) -> CandleDevice {
        panic!("Candle backend does not yet support interleaved complex tensors")
    }
}