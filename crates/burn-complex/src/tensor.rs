/*
Complex numbers
 */

pub struct ComplexTensor<B: Backend> {
    pub real: B::FloatTensorPrimitive,
    pub imag: B::FloatTensorPrimitive,
}

impl<B: Backend> TensorMetadata for ComplexTensor<B> {
    fn dtype(&self) -> burn_tensor::DType {
        self.primitive.dtype();
    }
    fn shape(&self) -> burn_tensor::Shape {
        self.primitive.shape();
    }
}
