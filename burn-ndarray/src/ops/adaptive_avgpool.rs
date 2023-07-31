use crate::{
    element::FloatNdArrayElement, iter_par, run_par, sharing::UnsafeSharedRef,
    tensor::NdArrayTensor,
};

use burn_tensor::ElementConversion;
use ndarray::Array4;

pub(crate) fn adaptive_avg_pool2d<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 4>,
    output_size: [usize; 2],
) -> NdArrayTensor<E, 4> {
    let [batch_size, channels, x_height, x_width] = x.shape().dims;

    let stride_h = f32::ceil(x_height as f32 / output_size[0] as f32) as usize;
    let stride_w = f32::ceil(x_width as f32 / output_size[1] as f32) as usize;
    let kernel_size_h = x_height - (output_size[0] - 1) * stride_h;
    let kernel_size_w = x_width - (output_size[1] - 1) * stride_w;
    let count = kernel_size_h * kernel_size_w;

    let x = x.array;
    let mut output = Array4::from_elem(
        (batch_size, channels, output_size[0], output_size[1]),
        0.elem(),
    );
    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    run_par!(|| {
        iter_par!(0, batch_size * channels).for_each(|k| unsafe {
            let b = k / channels;
            let c = k % channels;

            let output = unsafe_shared_out.get();
            for h in 0..output_size[0] {
                for w in 0..output_size[1] {
                    let mut sum_val: E = 0.elem();

                    for kh in 0..kernel_size_h {
                        let ih = h * stride_h + kh;

                        for kw in 0..kernel_size_w {
                            let iw = w * stride_w + kw;

                            if ih < x_height && iw < x_width {
                                sum_val += x[[b, c, ih, iw]];
                            }
                        }
                    }

                    output[[b, c, h, w]] = sum_val / (count as i32).elem();
                }
            }
        })
    });

    NdArrayTensor::new(output.into_dyn().into_shared())
}
