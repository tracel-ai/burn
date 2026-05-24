use burn_backend::{TypedDevice, UnimplementedTensorPrimitive};

use crate::{Candle, CandleDevice, FloatCandleElement, IntCandleElement};

impl<F: FloatCandleElement, I: IntCandleElement> TypedDevice<Candle<F,I>> for Candle<F, I> {
    fn complex_device(tensor: &UnimplementedTensorPrimitive<burn_std::Complex<F>>) -> CandleDevice {
        panic!("Candle backend does not yet support interleaved complex tensors")
    }
}