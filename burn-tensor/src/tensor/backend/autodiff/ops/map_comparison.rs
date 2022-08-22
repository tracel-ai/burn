use crate::backend::backend::Backend;
use crate::tensor::ops::*;

macro_rules! define_impl {
    (
        $backend:ty,
        $backend_inner:ty,
        $element:ident
    ) => {
        impl<E: $element, const D: usize> TensorOpsMapComparison<$backend, D>
            for <$backend as Backend>::TensorPrimitive<D>
        {
            fn equal(&self, other: &Self) -> <$backend as Backend>::BoolTensorPrimitive<D> {
                TensorOpsMapComparison::equal(&self.tensor(), &other.tensor())
            }

            fn equal_scalar(
                &self,
                other: &<$backend as Backend>::Elem,
            ) -> <$backend as Backend>::BoolTensorPrimitive<D> {
                TensorOpsMapComparison::equal_scalar(&self.tensor(), other)
            }

            fn greater(&self, other: &Self) -> <$backend as Backend>::BoolTensorPrimitive<D> {
                TensorOpsMapComparison::greater(&self.tensor(), &other.tensor())
            }

            fn greater_scalar(
                &self,
                other: &<$backend as Backend>::Elem,
            ) -> <$backend as Backend>::BoolTensorPrimitive<D> {
                TensorOpsMapComparison::greater_scalar(&self.tensor(), other)
            }

            fn greater_equal(&self, other: &Self) -> <$backend as Backend>::BoolTensorPrimitive<D> {
                TensorOpsMapComparison::greater_equal(&self.tensor(), &other.tensor())
            }

            fn greater_equal_scalar(
                &self,
                other: &<$backend as Backend>::Elem,
            ) -> <$backend as Backend>::BoolTensorPrimitive<D> {
                TensorOpsMapComparison::greater_equal_scalar(&self.tensor(), other)
            }

            fn lower(&self, other: &Self) -> <$backend as Backend>::BoolTensorPrimitive<D> {
                TensorOpsMapComparison::lower(&self.tensor(), &other.tensor())
            }

            fn lower_scalar(
                &self,
                other: &<$backend as Backend>::Elem,
            ) -> <$backend as Backend>::BoolTensorPrimitive<D> {
                TensorOpsMapComparison::lower_scalar(&self.tensor(), other)
            }

            fn lower_equal(&self, other: &Self) -> <$backend as Backend>::BoolTensorPrimitive<D> {
                TensorOpsMapComparison::lower_equal(&self.tensor(), &other.tensor())
            }

            fn lower_equal_scalar(
                &self,
                other: &<$backend as Backend>::Elem,
            ) -> <$backend as Backend>::BoolTensorPrimitive<D> {
                TensorOpsMapComparison::lower_equal_scalar(&self.tensor(), other)
            }
        }
    };
}

crate::register_tch!();
crate::register_ndarray!();
