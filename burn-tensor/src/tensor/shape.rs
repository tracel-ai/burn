use std::ops::Range;

#[derive(new, Debug, Clone, PartialEq, Eq, Copy)]
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

impl<const D1: usize> Shape<D1> {
    pub fn index<const D2: usize>(&self, indexes: [Range<usize>; D2]) -> Self {
        if D2 > D1 {
            panic!("Cant index that");
        }

        let mut dims = [0; D1];

        for i in 0..D2 {
            dims[i] = indexes[i].clone().count();
        }

        dims[D2..D1].copy_from_slice(&self.dims[D2..D1]);

        Self::new(dims)
    }

    pub fn remove_dim<const D2: usize>(&self, dim: usize) -> Shape<D2> {
        if D2 > D1 {
            panic!("Cant aggregate");
        }

        let mut dims = [0; D2];
        let mut index = 0;

        for i in 0..D1 {
            if i != dim {
                dims[index] = self.dims[i];
                index += 1;
            }
        }

        Shape::new(dims)
    }

    pub fn higher(&self, other: &Self) -> Self {
        let sum_self: usize = self.dims.iter().sum();
        let sum_other: usize = other.dims.iter().sum();

        if sum_self < sum_other {
            *other
        } else {
            *self
        }
    }
}
