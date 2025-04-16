use alloc::{format, string::String};
use core::marker::PhantomData;

use burn_tensor::{
    backend::Backend,
    quantization::{QTensorPrimitive, QuantizationScheme},
};

use super::{RouterTensor, RunnerChannel, RunnerClient, get_client, set_seed};

/// A backend that forwards the tensor operations to the appropriate backend (given multiple backends).
pub struct BackendRouter<R: RunnerChannel> {
    r: PhantomData<R>,
}

impl<R: RunnerChannel> core::fmt::Debug for BackendRouter<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("router"))
    }
}

impl<R: RunnerChannel> Clone for BackendRouter<R> {
    fn clone(&self) -> Self {
        Self { r: PhantomData }
    }
}

impl<R: RunnerChannel> Default for BackendRouter<R> {
    fn default() -> Self {
        Self { r: PhantomData }
    }
}

// TODO: quantization tensor primitive (w/ qparams)
impl<R: RunnerClient> QTensorPrimitive for RouterTensor<R> {
    fn scheme(&self) -> &QuantizationScheme {
        todo!()
    }
}

impl<R: RunnerChannel> Backend for BackendRouter<R> {
    type Device = R::Device;

    type FloatTensorPrimitive = RouterTensor<R::Client>;

    type FloatElem = R::FloatElem;

    type IntTensorPrimitive = RouterTensor<R::Client>;

    type IntElem = R::IntElem;

    type BoolTensorPrimitive = RouterTensor<R::Client>;

    type BoolElem = R::BoolElem;

    type QuantizedTensorPrimitive = RouterTensor<R::Client>;

    type QuantizedEncoding = u32;

    fn name(device: &Self::Device) -> String {
        format!("router<{}>", R::name(device))
    }

    fn seed(seed: u64) {
        set_seed(seed)
    }

    fn sync(device: &Self::Device) {
        let client = get_client::<R>(device);
        burn_common::future::block_on(client.sync());
    }
}
