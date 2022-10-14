use crate::backend::autodiff::ADBackendDecorator;
use crate::backend::Backend;
use crate::tensor::ops::*;

impl<B: Backend, const D: usize> TensorOpsMapComparison<ADBackendDecorator<B>, D>
    for <ADBackendDecorator<B> as Backend>::TensorPrimitive<D>
{
    fn equal(&self, other: &Self) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        TensorOpsMapComparison::equal(&self.tensor(), &other.tensor())
    }

    fn equal_scalar(
        &self,
        other: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        TensorOpsMapComparison::equal_scalar(&self.tensor(), other)
    }

    fn greater(&self, other: &Self) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        TensorOpsMapComparison::greater(&self.tensor(), &other.tensor())
    }

    fn greater_scalar(
        &self,
        other: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        TensorOpsMapComparison::greater_scalar(&self.tensor(), other)
    }

    fn greater_equal(
        &self,
        other: &Self,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        TensorOpsMapComparison::greater_equal(&self.tensor(), &other.tensor())
    }

    fn greater_equal_scalar(
        &self,
        other: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        TensorOpsMapComparison::greater_equal_scalar(&self.tensor(), other)
    }

    fn lower(&self, other: &Self) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        TensorOpsMapComparison::lower(&self.tensor(), &other.tensor())
    }

    fn lower_scalar(
        &self,
        other: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        TensorOpsMapComparison::lower_scalar(&self.tensor(), other)
    }

    fn lower_equal(
        &self,
        other: &Self,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        TensorOpsMapComparison::lower_equal(&self.tensor(), &other.tensor())
    }

    fn lower_equal_scalar(
        &self,
        other: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        TensorOpsMapComparison::lower_equal_scalar(&self.tensor(), other)
    }
}
