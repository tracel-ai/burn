use alloc::vec::Vec;

#[derive(new, Debug, Clone, PartialEq, Eq)]
pub struct Shape<const D: usize> {
    pub dims: [usize; D],
}

impl<const D: usize> Shape<D> {
    pub fn num_elements(&self) -> usize {
        let mut num_elements = 1;
        for i in 0..D {
            num_elements *= self.dims[i];
        }

        num_elements
    }
}

impl<const D: usize> From<[usize; D]> for Shape<D> {
    fn from(dims: [usize; D]) -> Self {
        Shape::new(dims)
    }
}

impl<const D: usize> From<Vec<i64>> for Shape<D> {
    fn from(shape: Vec<i64>) -> Self {
        let mut dims = [1; D];
        for (i, dim) in shape.into_iter().enumerate() {
            dims[i] = dim as usize;
        }
        Self::new(dims)
    }
}

impl<const D: usize> From<Vec<usize>> for Shape<D> {
    fn from(shape: Vec<usize>) -> Self {
        let mut dims = [1; D];
        for (i, dim) in shape.into_iter().enumerate() {
            dims[i] = dim;
        }
        Self::new(dims)
    }
}
