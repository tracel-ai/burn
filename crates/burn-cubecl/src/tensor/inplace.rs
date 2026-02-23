use std::sync::Arc;

use burn_backend::TensorMetadata;
use burn_std::{DType, Shape};

use crate::{CubeRuntime, tensor::CubeTensor};

// TODO: This is highly unsafe and inefficient. Should work with handles or whatever.
#[derive(Debug)]
pub struct CubeCommTensor<R: CubeRuntime>(pub Arc<*mut CubeTensor<R>>);

impl<R> From<&mut CubeTensor<R>> for CubeCommTensor<R>
where
    R: CubeRuntime,
{
    fn from(value: &mut CubeTensor<R>) -> Self {
        Self(Arc::new(value))
    }
}

impl<R> Clone for CubeCommTensor<R>
where
    R: CubeRuntime,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

unsafe impl<R> Sync for CubeCommTensor<R> where R: CubeRuntime {}
unsafe impl<R> Send for CubeCommTensor<R> where R: CubeRuntime {}

impl<R: CubeRuntime> TensorMetadata for CubeCommTensor<R> {
    fn dtype(&self) -> DType {
        unsafe { (**self.0).dtype() }
    }

    fn shape(&self) -> Shape {
        unsafe { (**self.0).shape() }
    }

    fn rank(&self) -> usize {
        unsafe { (**self.0).shape().num_dims() }
    }
}
