use super::{CubeType, ExpandElement, Tensor, UInt};

pub trait Vectorized {
    fn vectorization_factor(&self) -> UInt;
    fn vectorize(self, factor: UInt) -> Self;
}

impl<T: Vectorized + CubeType> Vectorized for Tensor<T> {
    fn vectorization_factor(&self) -> UInt {
        UInt::new(self.factor as u32)
    }

    fn vectorize(mut self, factor: UInt) -> Self {
        self.factor = factor.vectorization;
        self
    }
}

impl Vectorized for ExpandElement {
    fn vectorization_factor(&self) -> UInt {
        let var = match self {
            ExpandElement::Managed(var) => var,
            ExpandElement::Plain(var) => var,
        };

        UInt::new(var.item().vectorization as u32)
    }

    fn vectorize(self, _factor: UInt) -> Self {
        todo!()
    }
}
