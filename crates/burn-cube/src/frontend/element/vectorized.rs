use crate::unexpanded;

use super::{CubeType, ExpandElement, Tensor, UInt};

pub trait Vectorized {
    fn vectorization_factor(&self) -> UInt;
    fn vectorize(self, factor: UInt) -> Self;
}

impl<T: Vectorized + CubeType> Vectorized for Tensor<T> {
    fn vectorization_factor(&self) -> UInt {
        unexpanded!()
    }

    fn vectorize(self, _factor: UInt) -> Self {
        unexpanded!()
    }
}

impl<T: Vectorized + CubeType> Vectorized for &Tensor<T> {
    fn vectorization_factor(&self) -> UInt {
        unexpanded!()
    }

    fn vectorize(self, _factor: UInt) -> Self {
        unexpanded!()
    }
}

impl<T: Vectorized + CubeType> Vectorized for &mut Tensor<T> {
    fn vectorization_factor(&self) -> UInt {
        unexpanded!()
    }

    fn vectorize(self, _factor: UInt) -> Self {
        unexpanded!()
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

impl Vectorized for &ExpandElement {
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
