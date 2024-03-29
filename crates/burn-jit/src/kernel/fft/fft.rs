use crate::{
    compute::StaticKernel,
    element::JitElement,
    kernel::{self, build_info, elemwise_workgroup, KernelSettings, WORKGROUP_DEFAULT},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
};
use burn_tensor::Element;

kernel_wgsl!(FFT, "../../template/fft/fft.wgsl");

pub(crate) fn fft<R: Runtime, E: JitElement + Element>(
    input: JitTensor<R, E, 3>,
) -> JitTensor<R, E, 3> {

    let input = kernel::into_contiguous(input);

    let [_, num_samples, complex] = input.shape.dims;

    if complex != 2 {
        panic!("Last dimension must have size exactly 2");
    }

    // Power of 2 => only 1 bit set => x & (x - 1) == 0
    if num_samples == 0 || (num_samples & (num_samples - 1)) != 0 {
        panic!("Fourier transform dimension must have a power of 2 size, consider zero padding")
    };

    let output: JitTensor<R, E, 3> = empty_device(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );
    let mut info = build_info(&[&input, &output]);

    let num_elems = output.shape.num_elements();

    let num_fft_iters = (num_samples as f32).log2() as usize;
    info.push(num_fft_iters as u32);

    // Run the FFT
    info.push(0u32); // 0th iteration
    for fft_iter in 0..num_fft_iters {

        info.pop();
        info.push(fft_iter as u32);
        let info_handle = input.client.create(bytemuck::cast_slice(&info));

        // "Ping pong" buffering
        let (x_handle, x_hat_handle) = match fft_iter % 2 == 0 {
            true => (&input.handle, &output.handle),
            false => (&output.handle, &input.handle),
        };

        let kernel = StaticKernel::<
            KernelSettings<FFT, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
        >::new(elemwise_workgroup(num_elems, WORKGROUP_DEFAULT));
        input.client.execute(
            Box::new(kernel),
            &[
                x_handle,
                x_hat_handle,
                &info_handle,
            ],
        );
    }

    // "Ping pong" buffering
    let output = match num_fft_iters % 2 == 0 {
        true => input,
        false => output,
    };

    output
}
