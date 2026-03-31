use crate::{CubeRuntime, tensor::CubeTensor};
use cubecl::prelude::*;
//use cubecl::std::tensor::TensorHandle;
use crate::ops::numeric::empty_device_dtype;
use burn_backend::TensorMetadata;
use cubek::fft::rfft_launch;

//pub fn attention<R: CubeRuntime>(
//    query: CubeTensor<R>,
//    key: CubeTensor<R>,
//    value: CubeTensor<R>,
//    mask: Option<CubeTensor<R>>,
//    attn_bias: Option<CubeTensor<R>>,
//    options: AttentionModuleOptions,
//    strategy: AttentionStrategy,
//    out: Option<CubeTensor<R>>,
//) -> Result<CubeTensor<R>, AttentionSetupError> {

/// TODO
pub fn rfft<R: CubeRuntime>(signal: CubeTensor<R>, dim: usize) -> (CubeTensor<R>, CubeTensor<R>) {
    //let client = <R as Runtime>::client(&Default::default());
    //let device = Default::default();
    let dtype = f32::as_type_native_unchecked().storage_type();

    let signal_shape = signal.shape();
    let mut output_shape = signal_shape.clone();
    output_shape[dim] = output_shape[dim] / 2 + 1;

    let output_re = empty_device_dtype(
        signal.client.clone(),
        signal.device.clone(),
        output_shape.clone(),
        signal.dtype,
    );
    let output_im = empty_device_dtype(
        signal.client.clone(),
        signal.device.clone(),
        output_shape.clone(),
        signal.dtype,
    );

    rfft_launch(
        &signal.client.clone(),
        signal.binding(),
        output_re.clone().binding(),
        output_im.clone().binding(),
        dim,
        dtype,
    )
    .unwrap();

    (output_re, output_im)
}
