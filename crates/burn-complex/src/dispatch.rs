use burn_backend::BackendTypes;
use burn_tensor::backend::extension::Dispatch;

use crate::base::{CBT, ComplexTensorBackend, InterleavedLayout};

impl CBT for Dispatch {
    type ComplexTensorPrimitive;

    type ComplexScalar;
}
impl ComplexTensorBackend for Dispatch<B> {}
