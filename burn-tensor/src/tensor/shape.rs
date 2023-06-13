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

    pub fn fan_in(&self) -> usize {
        let receptive_field_size: usize = self.dims.iter().skip(2).product();
        *self
            .dims
            .get(1)
            .expect("Cannot get fan in of vector with dim < 2")
            * receptive_field_size
    }

    pub fn fan_out(&self) -> usize {
        let receptive_field_size: usize = self.dims.iter().skip(2).product();
        *self
            .dims
            .get(0)
            .expect("Cannot get fan in of vector with dim < 1")
            * receptive_field_size
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

    #[test]
    fn fan_in() {
        let shape1 = Shape::new([2, 3]);
        let shape2 = Shape::new([2, 3, 4, 5]);
        assert_eq!(3, shape1.fan_in());
        assert_eq!(60, shape2.fan_in());
    }

    #[test]
    fn fan_out() {
        let shape1 = Shape::new([2]);
        let shape2 = Shape::new([2, 3]);
        let shape3 = Shape::new([2, 3, 4, 5]);
        assert_eq!(2, shape1.fan_out());
        assert_eq!(2, shape2.fan_out());
        assert_eq!(40, shape3.fan_out());
    }
}
