use std::{f32::consts::PI, marker::PhantomData};

use crate::{
    codegen::{
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
        OutputInfo,
    },
    element::JitElement,
    gpu::{gpu, Branch, Elem, Scope, Variable, Visibility},
    kernel::{self, DynamicKernelSource, SourceTemplate},
    ops::numeric::empty_device,
    tensor::JitTensor,
    Compiler, Runtime,
};
use burn_tensor::Element;

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

        let kernel = FftEagerKernel::<R, E>::new();

        Execution::start(kernel, input.client.clone())
            .inputs(&[EagerHandle::<R>::new(
                &x_tensor.handle,
                &x_tensor.strides,
                &x_tensor.shape.dims,
            )])
            .outputs(&[EagerHandle::new(
                &x_hat_tensor.handle,
                &x_hat_tensor.strides,
                &x_hat_tensor.shape.dims,
            )])
            .with_scalars(&[num_fft_iters as u32, fft_iter as u32])
            .execute(crate::codegen::WorkgroupLaunch::Input { pos: 0 });
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

#[derive(new, Clone)]
pub(crate) struct FftEagerKernel<R: Runtime, E: JitElement> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for FftEagerKernel<R, E> {
    fn source(&self) -> SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let x = Variable::GlobalInputArray(0, item);
        let x_hat = Variable::GlobalOutputArray(0, item);

        FftComputeShader { x, x_hat }.expand(&mut scope);

        let x = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let x_hat = OutputInfo::Array { item };
        let scalars = InputInfo::Scalar {
            elem: Elem::UInt,
            size: 2,
        };

        let info = CompilationInfo {
            inputs: vec![x, scalars],
            outputs: vec![x_hat],
            scope,
        };

        let settings = CompilationSettings::default();
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>())
    }
}

pub(crate) struct FftComputeShader {
    x: Variable,
    x_hat: Variable,
}

impl FftComputeShader {
    fn expand(self, scope: &mut Scope) {
        let input = self.x;
        let output = self.x_hat;
        let id = Variable::Id;

        let input_stride_0 = scope.create_local(Elem::UInt);
        let input_stride_1 = scope.create_local(Elem::UInt);
        let input_stride_2 = scope.create_local(Elem::UInt);
        let input_shape_0 = scope.create_local(Elem::UInt);
        let input_shape_1 = scope.create_local(Elem::UInt);
        let input_shape_2 = scope.create_local(Elem::UInt);

        let output_stride_0 = scope.create_local(Elem::UInt);
        let output_stride_1 = scope.create_local(Elem::UInt);
        let output_stride_2 = scope.create_local(Elem::UInt);
        let output_shape_0 = scope.create_local(Elem::UInt);
        let output_shape_1 = scope.create_local(Elem::UInt);
        let output_shape_2 = scope.create_local(Elem::UInt);

        gpu!(scope, input_stride_0 = stride(input, 0u32));
        gpu!(scope, input_stride_1 = stride(input, 1u32));
        gpu!(scope, input_stride_2 = stride(input, 2u32));
        gpu!(scope, input_shape_0 = shape(input, 0u32));
        gpu!(scope, input_shape_1 = shape(input, 1u32));
        gpu!(scope, input_shape_2 = shape(input, 2u32));

        gpu!(scope, output_stride_0 = stride(output, 0u32));
        gpu!(scope, output_stride_1 = stride(output, 1u32));
        gpu!(scope, output_stride_2 = stride(output, 2u32));
        gpu!(scope, output_shape_0 = shape(output, 0u32));
        gpu!(scope, output_shape_1 = shape(output, 1u32));
        gpu!(scope, output_shape_2 = shape(output, 2u32));

        let num_fft_iters = Variable::GlobalScalar(0, Elem::UInt);
        let fft_iter = Variable::GlobalScalar(1, Elem::UInt);
        let fft_iter_plus = scope.create_with_value(1, Elem::UInt);
        gpu!(scope, fft_iter_plus += fft_iter);

        let is_final_iter = scope.create_local(Elem::Bool);
        gpu!(scope, is_final_iter = fft_iter_plus == num_fft_iters);

        // FFT is done over X dimension, parallelised over Y dimension. It's always
        //  a 1D transform, but many 1D transforms are done in parallel.
        let oy = scope.create_local(Elem::UInt);
        let ox = scope.create_local(Elem::UInt);
        let oc = scope.create_local(Elem::UInt);
        let iy = scope.create_local(Elem::UInt);

        gpu!(scope, oy = id / output_stride_0);
        gpu!(scope, oy = oy % output_shape_0);

        gpu!(scope, ox = id / output_stride_1);
        gpu!(scope, ox = ox % output_shape_1);

        gpu!(scope, oc = id / output_stride_2);
        gpu!(scope, oc = oc % output_shape_2);

        gpu!(scope, iy = oy);

        // Only ever 0 or 1 (real or imaginary). Arbitrarily choose the real index
        //  to do both complex calculations.
        let outside = scope.create_local(Elem::Bool);
        let outside_tmp = scope.create_local(Elem::Bool);
        gpu!(scope, outside = oc >= 1u32);
        gpu!(scope, if(outside).then(|scope| {
            scope.register(Branch::Return)
        }));

        // Number of independent FFTs at this stage (starts at x_width/2, halves each time)
        let num_transforms = scope.create_local(Elem::UInt);
        gpu!(scope, num_transforms = input_shape_1 >> fft_iter_plus);

        // Binary mask for extracting the index of E_k
        let even_mask = scope.create_local(Elem::UInt);
        let xffff = scope.create_with_value(65535, Elem::UInt);
        gpu!(scope, even_mask = num_transforms ^ xffff);

        // Returns if outside the output dimension
        gpu!(scope, outside = oy >= output_shape_0);
        gpu!(scope, outside_tmp = ox >= output_shape_1);
        gpu!(scope, outside = outside || outside_tmp);
        gpu!(scope, if(outside).then(|scope| {
            scope.register(Branch::Return)
        }));

        let ix_even = scope.create_local(Elem::UInt);
        let ix_odd = scope.create_local(Elem::UInt);
        gpu!(scope, ix_even = ox & even_mask);
        gpu!(scope, ix_odd = ix_even + num_transforms);

        let iter_diff = scope.create_local(Elem::UInt);
        gpu!(scope, iter_diff = num_fft_iters - fft_iter);
        gpu!(scope, iter_diff = ox >> iter_diff);
        let exponent = self.reverse_bits(scope, iter_diff, fft_iter);
        let exponent_float = scope.create_local(input.item());
        gpu!(scope, exponent_float = cast(exponent));

        let negative_instead_of_plus = scope.create_local(Elem::Bool);
        let ox_num = scope.create_local(Elem::UInt);
        gpu!(scope, ox_num = ox & num_transforms);
        gpu!(scope, negative_instead_of_plus = ox_num > 0u32);

        let i_even_re = scope.create_local(Elem::UInt);
        let i_even_im = scope.create_local(Elem::UInt);
        let i_odd_re = scope.create_local(Elem::UInt);
        let i_odd_im = scope.create_local(Elem::UInt);
        let i_out_re = scope.create_local(Elem::UInt);
        let i_out_im = scope.create_local(Elem::UInt);

        let iy_ = scope.create_local(Elem::UInt);
        let ix_even_ = scope.create_local(Elem::UInt);
        let ix_odd_ = scope.create_local(Elem::UInt);
        let oy_ = scope.create_local(Elem::UInt);
        let ox_r_ = scope.create_local(Elem::UInt);

        // Running the FFT algorithm like this results in a bit-reversed ordered
        //  output. i.e. the element 000, 001, ..., 110, 111 are now sorted if
        //  they were actually 000, 100, ..., 011, 111. On the last step, undo this
        //  mapping, by choosing ox differently.

        let ox_r = scope.create_local(Elem::UInt);
        gpu!(scope, if(is_final_iter).then(|scope|{
            let reversed = self.reverse_bits(scope, ox, num_fft_iters);
            gpu!(scope, ox_r = reversed);
        }).else(|scope|{
            gpu!(scope, ox_r = ox);
        }));

        gpu!(scope, iy_ = iy * input_stride_0);
        gpu!(scope, ix_even_ = ix_even * input_stride_1);
        gpu!(scope, ix_odd_ = ix_odd * input_stride_1);

        gpu!(scope, oy_ = oy * output_stride_0);
        gpu!(scope, ox_r_ = ox_r * output_stride_1);

        gpu!(scope, i_even_re = iy_ + ix_even_);
        gpu!(scope, i_even_im = i_even_re + input_stride_2);

        gpu!(scope, i_odd_re = iy_ + ix_odd_);
        gpu!(scope, i_odd_im = i_odd_re + input_stride_2);

        gpu!(scope, i_out_re = oy_ + ox_r_);
        gpu!(scope, i_out_im = i_out_re + output_stride_2);

        // Here we compute the main computation step for each index.
        //  See the last two equations of:
        //  https://en.wikipedia.org/wiki/Cooley-Tukey_FFT_algorithm#The_radix-2_DIT_case
        // X_k = E_k + w_k * O_k
        //  Where w_k is the +/- exp(-2 pi i k / n) term. Note the plus / minus
        //  is included in the value of the weight.

        let pm1 = scope.create_local(input.item());
        gpu!(scope, if(negative_instead_of_plus).then(|scope|{
            let tmp = scope.create_with_value(-1, input.item());
            gpu!(scope, pm1 = tmp);
        }).else(|scope|{
            let tmp = scope.create_with_value(1, input.item());
            gpu!(scope, pm1 = tmp);
        }));

        // Width of the FFT at this stage (starts at 2, doubles each time)
        let two = scope.create_with_value(2, Elem::UInt);
        let n = scope.create_local(Elem::UInt);
        let n_float = scope.create_local(input.item());
        gpu!(scope, n = two << fft_iter);
        gpu!(scope, n_float = cast(n));

        let pi = scope.create_with_value(PI, input.item());
        let w_k_theta = scope.create_with_value(-2, input.item());
        gpu!(scope, w_k_theta *= pi);
        gpu!(scope, w_k_theta *= exponent_float);
        gpu!(scope, w_k_theta = w_k_theta / n_float);

        let w_k_re = scope.create_local(input.item());
        let w_k_im = scope.create_local(input.item());
        gpu!(scope, w_k_re = cos(w_k_theta));
        gpu!(scope, w_k_im = sin(w_k_theta));
        gpu!(scope, w_k_re *= pm1);
        gpu!(scope, w_k_im *= pm1);

        let e_k_re = scope.create_local(input.item());
        let e_k_im = scope.create_local(input.item());
        let o_k_re = scope.create_local(input.item());
        let o_k_im = scope.create_local(input.item());

        gpu!(scope, e_k_re = input[i_even_re]);
        gpu!(scope, e_k_im = input[i_even_im]);
        gpu!(scope, o_k_re = input[i_odd_re]);
        gpu!(scope, o_k_im = input[i_odd_im]);

        // Note the following:
        // Real part of (a + bj)(c + dj) = ac + bd(j*j) = ac - bd
        // Imaginary part of (a + bj)(c + dj) = ad(j) + bc(j) = (ad + bc)j
        // These are used for w_k * O_k; E_k real and imaginary parts are just added.
        let out_re_value = scope.create_local(output.item());
        let out_im_value = scope.create_local(output.item());
        let out_value_tmp = scope.create_local(output.item());

        gpu!(scope, out_value_tmp = w_k_im * o_k_im);
        gpu!(scope, out_re_value = w_k_re * o_k_re);
        gpu!(scope, out_re_value += e_k_re);
        gpu!(scope, out_re_value = out_re_value - out_value_tmp);

        gpu!(scope, out_value_tmp = w_k_im * o_k_re);
        gpu!(scope, out_im_value = w_k_re * o_k_im);
        gpu!(scope, out_im_value += e_k_im);
        gpu!(scope, out_im_value = out_im_value + out_value_tmp);

        gpu!(scope, output[i_out_re] = out_re_value);
        gpu!(scope, output[i_out_im] = out_im_value);
    }

    fn reverse_bits(&self, scope: &mut Scope, x_: Variable, num_bits: Variable) -> Variable {
        // This function assumes UInt is u32
        let n_bits = scope.create_with_value(32, Elem::UInt);

        // For input:
        // 00000000000000000000000000011011
        //  num_bits = 1 gives:
        // 00000000000000000000000000000001
        //  num_bits = 2:
        // 00000000000000000000000000000011
        //  num_bits = 3:
        // 00000000000000000000000000000110
        //  num_bits = 4:
        // 00000000000000000000000000001101
        //  etc...

        let new_x = scope.create_local(Elem::UInt);
        gpu!(scope, new_x = reverseBits(x_));

        let shift = scope.create_local(Elem::UInt);
        gpu!(scope, shift = min(n_bits, num_bits));
        gpu!(scope, shift = n_bits - shift);
        gpu!(scope, new_x = new_x >> shift);

        new_x
    }
}
