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
        panic!("Last dimension must have size exactly 2 (real, imaginary)");
    }

    // Power of 2 => only 1 bit set => x & (x - 1) == 0
    if num_samples == 0 || (num_samples & (num_samples - 1)) != 0 {
        panic!("Fourier transform dimension must have a power of 2 size, perhaps consider zero padding")
    };

    // Need to use two output buffers as the algorithm writes back and forth
    //  at each iteration. We could reuse the input buffer but this would
    //  modify in place which might be undesirable.
    let output_1: JitTensor<R, E, 3> = empty_device(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );
    let output_2: JitTensor<R, E, 3> = empty_device(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );

    let num_elems = input.shape.num_elements();
    let num_fft_iters = (num_samples as f32).log2() as usize;

    for fft_iter in 0..num_fft_iters {
        // "Ping pong" buffering
        let (x_tensor, x_hat_tensor) = {
            if fft_iter == 0 {
                (&input, &output_1)
            } else if fft_iter % 2 == 0 {
                (&output_2, &output_1)
            } else {
                (&output_1, &output_2)
            }
        };

        let mut info = build_info(&[&x_tensor, &x_hat_tensor]);
        info.push(num_fft_iters as u32);
        info.push(fft_iter as u32);

        let info_handle = input.client.create(bytemuck::cast_slice(&info));

        let kernel = StaticKernel::<
            KernelSettings<FFT, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
        >::new(elemwise_workgroup(num_elems, WORKGROUP_DEFAULT));
        input.client.execute(
            Box::new(kernel),
            &[&x_tensor.handle, &x_hat_tensor.handle, &info_handle],
        );
    }

    // "Ping pong" buffering
    {
        if num_fft_iters == 0 {
            input
        } else if num_fft_iters % 2 == 0 {
            output_2
        } else {
            output_1
        }
    }
}
