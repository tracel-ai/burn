use alloc::{format, string::String};
use core::marker::PhantomData;

use burn_tensor::{
    backend::{Backend, BackendBridge},
    ops::FloatTensor,
    quantization::{QTensorPrimitive, QuantizationScheme, QuantizationStrategy},
    repr::{BaseOperationDescription, OperationDescription, UnaryOperationDescription},
    Device, Element,
};

use super::{get_client, set_seed, RouterTensor, RunnerChannel, RunnerClient};

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

    fn strategy(&self) -> QuantizationStrategy {
        todo!()
    }
}

impl<R: RunnerChannel> Backend for BackendRouter<R> {
    type Device = R::Device;

    type FullPrecisionBridge = PrecisionBridge;

    type FloatTensorPrimitive = RouterTensor<R::Client>;

    type FloatElem = R::FloatElem;

    type IntTensorPrimitive = RouterTensor<R::Client>;

    type IntElem = R::IntElem;

    type BoolTensorPrimitive = RouterTensor<R::Client>;

    type BoolElem = R::BoolElem;

    type QuantizedTensorPrimitive = RouterTensor<R::Client>;

    type QuantizedEncoding = u32;

    fn name() -> String {
        format!("router<{}>", R::name())
    }

    fn seed(seed: u64) {
        set_seed(seed)
    }

    fn sync(device: &Self::Device) {
        let client = get_client::<R>(device);
        burn_common::future::block_on(client.sync());
    }
}

/// Handle precision conversion.
#[derive(Debug)]
pub struct PrecisionBridge {}

impl<R: RunnerChannel> BackendBridge<BackendRouter<R>> for PrecisionBridge {
    type Target = BackendRouter<R>;

    fn into_target(
        tensor: FloatTensor<BackendRouter<R>>,
        _device: Option<Device<Self::Target>>,
    ) -> FloatTensor<Self::Target> {
        let client = tensor.client.clone();
        let out = client.register_float_tensor(
            tensor.shape.clone(),
            <Self::Target as Backend>::FloatElem::dtype().into(),
        );

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::BaseFloat(
            BaseOperationDescription::Cast(desc),
        ));

        out
    }

    fn from_target(
        tensor: FloatTensor<Self::Target>,
        _device: Option<Device<BackendRouter<R>>>,
    ) -> FloatTensor<BackendRouter<R>> {
        let client = tensor.client.clone();
        let out = client.register_float_tensor(tensor.shape.clone(), R::FloatElem::dtype().into());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::BaseFloat(
            BaseOperationDescription::Cast(desc),
        ));

        out
    }
}
