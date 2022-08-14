use crate::{back::Backend, tensor::ops::*};
use rand::distributions::Standard;

macro_rules! define_impl {
    (
        $backend:ty,
        $backend_inner:ty,
        $element:ident
    ) => {
        impl<E: $element, const D: usize> TensorOpsAggregation<$backend, D>
            for <$backend as Backend>::TensorPrimitive<D>
        where
            Standard: rand::distributions::Distribution<E>,
        {
            fn mean(&self) -> <$backend as Backend>::TensorPrimitive<1> {
                todo!()
            }

            fn sum(&self) -> <$backend as Backend>::TensorPrimitive<1> {
                todo!()
            }

            fn mean_dim<const D2: usize>(
                &self,
                dim: usize,
            ) -> <$backend as Backend>::TensorPrimitive<D2> {
                todo!()
            }

            fn sum_dim<const D2: usize>(
                &self,
                dim: usize,
            ) -> <$backend as Backend>::TensorPrimitive<D2> {
                todo!()
            }

            fn mean_dim_keepdim(&self, dim: usize) -> <$backend as Backend>::TensorPrimitive<D> {
                todo!()
            }

            fn sum_dim_keepdim(&self, dim: usize) -> <$backend as Backend>::TensorPrimitive<D> {
                todo!()
            }
        }
    };
}

crate::register_tch!();
crate::register_ndarray!();
