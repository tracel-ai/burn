use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};
use burn_tensor::Shape;
use cubecl::{CubeCountSettings, Execution};

use super::AdaptivePool2dEagerKernel;

pub(crate) fn adaptive_avg_pool2d<R: JitRuntime, E: JitElement>(
    input: JitTensor<R, E>,
    output_size: [usize; 2],
) -> JitTensor<R, E> {
    let [batch_size, channels, _, _] = input.shape.dims();

    let output_shape = Shape::new([batch_size, channels, output_size[0], output_size[1]]);
    let output = empty_device(input.client.clone(), input.device.clone(), output_shape);

    let kernel = AdaptivePool2dEagerKernel::<R, E>::new();

    Execution::start(kernel, input.client.clone())
        .inputs(&[input.as_handle_ref()])
        .outputs(&[output.as_handle_ref()])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}
