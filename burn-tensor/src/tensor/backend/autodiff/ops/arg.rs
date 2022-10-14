use crate::backend::autodiff::ADBackendDecorator;
use crate::backend::Backend;
use crate::tensor::ops::*;

impl<B: Backend, const D: usize> TensorOpsArg<ADBackendDecorator<B>, D>
    for <ADBackendDecorator<B> as Backend>::TensorPrimitive<D>
{
    fn argmax(
        &self,
        dim: usize,
    ) -> <<ADBackendDecorator<B> as Backend>::IntegerBackend as Backend>::TensorPrimitive<D> {
        TensorOpsArg::argmax(&self.tensor(), dim)
    }

    fn argmin(
        &self,
        dim: usize,
    ) -> <<ADBackendDecorator<B> as Backend>::IntegerBackend as Backend>::TensorPrimitive<D> {
        TensorOpsArg::argmin(&self.tensor(), dim)
    }
}
