//! F32 implicit convolution candidates for devices without a suitable matrix accelerator.

use burn_backend::{
    DType,
    cubecl::dtype_to_storage_type,
    ops::{ConvOptions, conv::calculate_conv_output_sizes},
};
use burn_std::Shape;
use cubek::{
    convolution::{
        ConvolutionArgs, ConvolutionInputs, UnitStrategy, components::ConvSetupError, launch_unit,
    },
    matmul::definition::MatmulElems,
    std::InputBinding,
};

use super::{
    backward_data::fallback::conv_data_backward_fallback,
    backward_weight::fallback::conv_weight_backward_fallback, conv_direct,
};
use crate::{ops::numeric::empty_device_dtype, tensor::CubeTensor};

fn check<const N: usize>(
    input: &CubeTensor,
    kernel: &[usize],
    options: &ConvOptions<N>,
) -> Result<(), ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }
    // Eligibility depends on exact key fields and on the device. Spatial
    // dimensions and channel counts are bucketed and must not cause a decline.
    if input.dtype != DType::F32
        || !(1..=3).contains(&N)
        || kernel.iter().product::<usize>() > 49
        || input.client.properties().hardware.num_cpu_cores.is_some()
    {
        return Err(ConvSetupError::Unknown);
    }
    Ok(())
}

fn binding(input: &CubeTensor) -> InputBinding {
    InputBinding::new(input.clone().binding(), dtype_to_storage_type(input.dtype))
}

fn args<const N: usize>(options: &ConvOptions<N>) -> ConvolutionArgs<N> {
    ConvolutionArgs {
        stride: options.stride,
        padding: options.padding_begin(),
        dilation: options.dilation,
    }
}

pub(crate) fn conv_unit<const N: usize>(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvOptions<N>,
    strategy: UnitStrategy,
) -> Result<CubeTensor, ConvSetupError> {
    check(&input, &weight.meta.shape()[1..N + 1], &options)?;
    let mut shape = calculate_conv_output_sizes(
        &weight.meta.shape()[1..N + 1],
        &options.stride,
        &options.padding,
        &options.dilation,
        &input.meta.shape()[1..N + 1],
    );
    shape.insert(0, input.meta.shape()[0]);
    shape.push(weight.meta.shape()[0]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape.into(),
        input.dtype,
    );
    let result = launch_unit(
        strategy,
        &input.client,
        ConvolutionInputs::Forward {
            input: binding(&input),
            weight: binding(&weight),
            bias: bias.as_ref().map(binding),
            out: output.clone().binding(),
        },
        args(&options),
        MatmulElems::from_single_dtype(dtype_to_storage_type(input.dtype)),
    );
    if result.is_ok() {
        return Ok(output);
    }
    // A cached candidate can see a different shape/layout in the same bucket.
    // Keep a fully general fallback when that shape cannot use this tiling.
    conv_direct(input, weight, bias, options)
}

pub(crate) fn dgrad_unit<const N: usize>(
    out_grad: CubeTensor,
    weight: CubeTensor,
    shape: Shape,
    options: ConvOptions<N>,
    strategy: UnitStrategy,
) -> Result<CubeTensor, ConvSetupError> {
    check(&out_grad, &weight.meta.shape()[1..N + 1], &options)?;
    let output = empty_device_dtype(
        out_grad.client.clone(),
        out_grad.device.clone(),
        shape.clone(),
        out_grad.dtype,
    );
    let result = launch_unit(
        strategy,
        &out_grad.client,
        ConvolutionInputs::BackwardData {
            out_grad: binding(&out_grad),
            weights: binding(&weight),
            in_grad: output.clone().binding(),
        },
        args(&options),
        MatmulElems::from_single_dtype(dtype_to_storage_type(out_grad.dtype)),
    );
    if result.is_ok() {
        return Ok(output);
    }
    conv_data_backward_fallback(out_grad, weight, shape, options)
}

pub(crate) fn wgrad_unit<const N: usize>(
    input: CubeTensor,
    out_grad: CubeTensor,
    shape: Shape,
    options: ConvOptions<N>,
    strategy: UnitStrategy,
) -> Result<CubeTensor, ConvSetupError> {
    check(&input, &shape[1..N + 1], &options)?;
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape.clone(),
        input.dtype,
    );
    let result = launch_unit(
        strategy,
        &input.client,
        ConvolutionInputs::BackwardWeight {
            input: binding(&input),
            out_grad: binding(&out_grad),
            weight_grad: output.clone().binding(),
        },
        args(&options),
        MatmulElems::from_single_dtype(dtype_to_storage_type(input.dtype)),
    );
    if result.is_ok() {
        return Ok(output);
    }
    conv_weight_backward_fallback(input, out_grad, shape, options)
}
