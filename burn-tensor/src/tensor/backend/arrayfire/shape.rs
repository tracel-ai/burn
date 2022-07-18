use crate::Shape;
use arrayfire::Dim4;

impl<const D: usize> Into<Dim4> for Shape<D> {
    fn into(self) -> Dim4 {
        if D > 4 {
            panic!(
                "Can't create arrayfire Tensor with more than 4 dimensions, got {}",
                D
            );
        }
        let mut dims = [1; 4];
        for i in 0..D {
            dims[i] = self.dims[i] as u64;
        }
        Dim4::new(&dims)
    }
}

impl<const D: usize> From<Dim4> for Shape<D> {
    fn from(dim: Dim4) -> Self {
        let mut values = [0; D];
        for i in 0..D {
            values[i] = dim[i] as usize;
        }
        Shape::new(values)
    }
}
