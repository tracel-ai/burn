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
    let [batch_size, channels, input_height, input_width] = x.shape().dims;

    let x = x.array;
    let mut output = Array4::from_elem(
        (batch_size, channels, output_size[0], output_size[1]),
        0.elem(),
    );
    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    let start_index = |output_size_index: usize, output_size: usize, input_size: usize| {
        f32::floor((output_size_index as f32 * input_size as f32) / output_size as f32) as usize
    };

    let end_index = |output_size_index: usize, output_size: usize, input_size: usize| {
        f32::ceil(((output_size_index + 1) as f32 * input_size as f32) / output_size as f32)
            as usize
    };

    run_par!(|| {
        iter_par!(0, batch_size * channels).for_each(|k| unsafe {
            let b = k / channels;
            let c = k % channels;

            let output = unsafe_shared_out.get();
            for h in 0..output_size[0] {
                for w in 0..output_size[1] {
                    let ih_start = start_index(h, output_size[0], input_height);
                    let ih_end = end_index(h, output_size[0], input_height);
                    let iw_start = start_index(w, output_size[1], input_width);
                    let iw_end = end_index(w, output_size[1], input_width);

                    let mut count: i32 = 0;
                    let mut sum_val: E = 0.elem();

                    for ih in ih_start..ih_end {
                        for iw in iw_start..iw_end {
                            if ih < input_height && iw < input_width {
                                sum_val += x[[b, c, ih, iw]];
                                count += 1;
                            }
                        }
                    }

                    output[[b, c, h, w]] = sum_val / count.elem();
                }
            }
        })
    });

    NdArrayTensor::new(output.into_dyn().into_shared())
}
