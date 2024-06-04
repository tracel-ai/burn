use super::{CubeType, ExpandElement, Tensor};

pub trait Vectorized {
    fn vectorization_factor(&self) -> u8;
    fn vectorize(self, factor: u8) -> Self;
}

impl<T: Vectorized + CubeType> Vectorized for Tensor<T> {
    fn vectorization_factor(&self) -> u8 {
        self.factor
    }

    fn vectorize(mut self, factor: u8) -> Self {
        self.factor = factor;
        self
    }
}

impl Vectorized for ExpandElement {
    fn vectorization_factor(&self) -> u8 {
        let var = match self {
            ExpandElement::Managed(var) => var,
            ExpandElement::Plain(var) => var,
        };

        var.item().vectorization
    }

    fn vectorize(self, _factor: u8) -> Self {
        todo!()
    }
}
