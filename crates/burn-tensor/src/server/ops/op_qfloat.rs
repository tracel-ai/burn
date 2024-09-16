use core::future::Future;

use crate::{
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    quantization::{QuantizationParametersPrimitive, QuantizationScheme},
    server::{Server, ServerBackend},
    Device, TensorData,
};

impl<B: ServerBackend> QTensorOps<Self> for Server<B> {}
