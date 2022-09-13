use crate::backend::Backend;
use crate::tensor::ops::*;

macro_rules! define_impl {
    (
        $backend:ty,
        $backend_inner:ty,
        $element:ident
    ) => {
        impl<E: $element, const D: usize> TensorOpsArg<$backend, D>
            for <$backend as Backend>::TensorPrimitive<D>
        {
            fn argmax(
                &self,
                dim: usize,
            ) -> <<$backend as Backend>::IntegerBackend as Backend>::TensorPrimitive<D> {
                TensorOpsArg::argmax(&self.tensor(), dim)
            }

            fn argmin(
                &self,
                dim: usize,
            ) -> <<$backend as Backend>::IntegerBackend as Backend>::TensorPrimitive<D> {
                TensorOpsArg::argmin(&self.tensor(), dim)
            }
        }
    };
}

crate::register_tch!();
crate::register_ndarray!();
