use burn_core as burn;
use burn_core::record::Record;

use burn_tensor::backend::Backend;
use burn_tensor::Tensor;

// It compiles
#[derive(Record)]
pub struct TestWithBackendRecord<B: Backend> {
    tensor: Tensor<B, 2>,
}

// It compiles
#[derive(Record)]
pub struct TestWithoutBackendRecord {
    _tensor: usize,
}
