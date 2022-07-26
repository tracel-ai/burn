use crate::{backend::ndarray::NdArrayTensor, TensorOpsTranspose};

impl<P, const D: usize> TensorOpsTranspose<P, D> for NdArrayTensor<P, D>
where
    P: Clone + std::fmt::Debug,
{
    fn transpose(&self) -> Self {
        let array = self.array.t().into_dyn().into_owned().into_shared();
        let mut shape = self.shape.clone();

        if D >= 2 {
            let size0 = shape.dims[D - 2];
            let size1 = shape.dims[D - 1];
            shape.dims[D - 2] = size1;
            shape.dims[D - 1] = size0;
        }

        Self { array, shape }
    }
}
