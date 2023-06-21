use alloc::vec::Vec;

/// Shape of a tensor.
#[derive(new, Debug, Clone, PartialEq, Eq)]
pub struct Shape<const D: usize> {
    /// The dimensions of the tensor.
    pub dims: [usize; D],
}

impl<const D: usize> Shape<D> {
    /// Returns the total number of elements of a tensor having this shape
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

impl<const D: usize> From<Vec<u64>> for Shape<D> {
    fn from(shape: Vec<u64>) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_elements() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        assert_eq!(120, shape.num_elements());
    }
}
