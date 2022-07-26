use std::ops::Range;

#[derive(new, Debug, Clone, PartialEq, Copy)]
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

impl<const D1: usize> Shape<D1> {
    pub fn index<const D2: usize>(&self, indexes: [Range<usize>; D2]) -> Self {
        if D2 > D1 {
            panic!("Cant index that");
        }

        let mut dims = [0; D1];

        for i in 0..D2 {
            dims[i] = indexes[i].clone().count();
        }

        for i in D2..D1 {
            dims[i] = self.dims[i];
        }

        Self::new(dims)
    }
}
