use cubecl::{
    cpa,
    ir::{Elem, KernelDefinition, Scope, Variable, Visibility},
    CubeCountSettings, Execution, InputInfo, KernelExpansion, KernelIntegrator, KernelSettings,
    OutputInfo,
};
use std::marker::PhantomData;

use crate::{kernel::Kernel, tensor::JitTensor, JitElement, JitRuntime};

#[derive(new)]
struct InterpolateBicubicEagerKernel<R, E> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct InterpolateBicubicShader<E> {
    input: Variable,
    output: Variable,
    _elem: PhantomData<E>,
}

impl<E: JitElement> InterpolateBicubicShader<E> {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let input = self.input;
        let output = self.output;
        let id = Variable::AbsolutePos;
        let elem = E::cube_elem();

        let input_stride_0 = scope.create_local(Elem::UInt);
        let input_stride_1 = scope.create_local(Elem::UInt);
        let input_stride_2 = scope.create_local(Elem::UInt);
        let input_stride_3 = scope.create_local(Elem::UInt);

        let input_shape_2 = scope.create_local(Elem::UInt);
        let input_shape_3 = scope.create_local(Elem::UInt);

        let output_stride_0 = scope.create_local(Elem::UInt);
        let output_stride_1 = scope.create_local(Elem::UInt);
        let output_stride_2 = scope.create_local(Elem::UInt);
        let output_stride_3 = scope.create_local(Elem::UInt);

        let output_shape_0 = scope.create_local(Elem::UInt);
        let output_shape_1 = scope.create_local(Elem::UInt);
        let output_shape_2 = scope.create_local(Elem::UInt);
        let output_shape_3 = scope.create_local(Elem::UInt);

        cpa!(scope, input_stride_0 = stride(input, 0u32));
        cpa!(scope, input_stride_1 = stride(input, 1u32));
        cpa!(scope, input_stride_2 = stride(input, 2u32));
        cpa!(scope, input_stride_3 = stride(input, 3u32));

        cpa!(scope, input_shape_2 = shape(input, 2u32));
        cpa!(scope, input_shape_3 = shape(input, 3u32));

        cpa!(scope, output_stride_0 = stride(output, 0u32));
        cpa!(scope, output_stride_1 = stride(output, 1u32));
        cpa!(scope, output_stride_2 = stride(output, 2u32));
        cpa!(scope, output_stride_3 = stride(output, 3u32));

        cpa!(scope, output_shape_0 = shape(output, 0u32));
        cpa!(scope, output_shape_1 = shape(output, 1u32));
        cpa!(scope, output_shape_2 = shape(output, 2u32));
        cpa!(scope, output_shape_3 = shape(output, 3u32));

        let b = scope.create_local(Elem::UInt);
        let c = scope.create_local(Elem::UInt);
        let h = scope.create_local(Elem::UInt);
        let w = scope.create_local(Elem::UInt);

        cpa!(scope, b = id / output_stride_0);
        cpa!(scope, b = b % output_shape_0);

        cpa!(scope, c = id / output_stride_1);
        cpa!(scope, c = c % output_shape_1);

        cpa!(scope, h = id / output_stride_2);
        cpa!(scope, h = h % output_shape_2);

        cpa!(scope, w = id / output_stride_3);
        cpa!(scope, w = w % output_shape_3);

        let input_height = scope.create_local(Elem::UInt);
        let output_height = scope.create_local(Elem::UInt);
        let output_height_float = scope.create_local(elem);

        let input_width = scope.create_local(Elem::UInt);
        let output_width = scope.create_local(Elem::UInt);
        let output_width_float = scope.create_local(elem);

        let frac = scope.create_local(elem);
        let numerator = scope.create_local(Elem::UInt);
        let numerator_float = scope.create_local(elem);
        let not_zero = scope.create_local(Elem::Bool);

        let y_in_float = scope.create_local(elem);
        let y_in = scope.create_local(Elem::UInt);
        let yw = scope.create_local(elem);
        let y_tmp = scope.create_local(Elem::UInt);

        cpa!(scope, input_height = input_shape_2 - 1u32);
        cpa!(scope, output_height = output_shape_2 - 1u32);
        cpa!(scope, output_height = max(output_height, 1u32));
        cpa!(scope, numerator = h * input_height);
        cpa!(scope, numerator_float = cast(numerator));
        cpa!(scope, output_height_float = cast(output_height));
        cpa!(scope, frac = numerator_float / output_height_float);
        cpa!(scope, y_in_float = floor(frac));
        cpa!(scope, y_in = cast(y_in_float));
        cpa!(scope, yw = frac - y_in_float);

        let y0 = scope.zero(Elem::UInt);
        cpa!(scope, not_zero = y_in != 0u32);
        cpa!(scope, if(not_zero).then(|scope|{
            cpa!(scope, y0 = y_in - 1u32);
        }));

        let y1 = y_in;

        cpa!(scope, y_tmp = y_in + 1u32);
        let y2 = Self::min(scope, y_tmp, input_height);

        cpa!(scope, y_tmp = y_in + 2u32);
        let y3 = Self::min(scope, y_tmp, input_height);

        let x_in_float = scope.create_local(elem);
        let x_in = scope.create_local(Elem::UInt);
        let xw = scope.create_local(elem);
        let x_tmp = scope.create_local(Elem::UInt);

        cpa!(scope, input_width = input_shape_3 - 1u32);
        cpa!(scope, output_width = output_shape_3 - 1u32);
        cpa!(scope, output_width = max(output_width, 1u32));
        cpa!(scope, numerator = w * input_width);
        cpa!(scope, numerator_float = cast(numerator));
        cpa!(scope, output_width_float = cast(output_width));
        cpa!(scope, frac = numerator_float / output_width_float);
        cpa!(scope, x_in_float = floor(frac));
        cpa!(scope, x_in = cast(x_in_float));
        cpa!(scope, xw = frac - x_in_float);

        let x0 = scope.zero(Elem::UInt);
        cpa!(scope, not_zero = x_in != 0u32);
        cpa!(scope, if(not_zero).then(|scope|{
            cpa!(scope, x0 = x_in - 1u32);
        }));

        cpa!(scope, x_tmp = x_in - 1u32);
        let x1 = x_in;

        cpa!(scope, x_tmp = x_in + 1u32);
        let x2 = Self::min(scope, x_tmp, input_width);

        cpa!(scope, x_tmp = x_in + 2u32);
        let x3 = Self::min(scope, x_tmp, input_width);

        let index_base = scope.create_local(Elem::UInt);
        let index_tmp = scope.create_local(Elem::UInt);
        cpa!(scope, index_base = b * input_stride_0);
        cpa!(scope, index_tmp = c * input_stride_1);
        cpa!(scope, index_base += index_tmp);

        let y0_stride = scope.create_local(Elem::UInt);
        let y1_stride = scope.create_local(Elem::UInt);
        let y2_stride = scope.create_local(Elem::UInt);
        let y3_stride = scope.create_local(Elem::UInt);
        let x0_stride = scope.create_local(Elem::UInt);
        let x1_stride = scope.create_local(Elem::UInt);
        let x2_stride = scope.create_local(Elem::UInt);
        let x3_stride = scope.create_local(Elem::UInt);
        cpa!(scope, y0_stride = y0 * input_stride_2);
        cpa!(scope, y1_stride = y1 * input_stride_2);
        cpa!(scope, y2_stride = y2 * input_stride_2);
        cpa!(scope, y3_stride = y3 * input_stride_2);
        cpa!(scope, x0_stride = x0 * input_stride_3);
        cpa!(scope, x1_stride = x1 * input_stride_3);
        cpa!(scope, x2_stride = x2 * input_stride_3);
        cpa!(scope, x3_stride = x3 * input_stride_3);

        let index_0 = scope.create_local(Elem::UInt);
        let index_1 = scope.create_local(Elem::UInt);
        let index_2 = scope.create_local(Elem::UInt);
        let index_3 = scope.create_local(Elem::UInt);
        let inp_0 = scope.create_local(input.item());
        let inp_1 = scope.create_local(input.item());
        let inp_2 = scope.create_local(input.item());
        let inp_3 = scope.create_local(input.item());

        cpa!(scope, index_0 = index_base);
        cpa!(scope, index_0 += y0_stride);
        cpa!(scope, index_0 += x0_stride);
        cpa!(scope, inp_0 = input[index_0]);
        cpa!(scope, index_1 = index_base);
        cpa!(scope, index_1 += y0_stride);
        cpa!(scope, index_1 += x1_stride);
        cpa!(scope, inp_1 = input[index_1]);
        cpa!(scope, index_2 = index_base);
        cpa!(scope, index_2 += y0_stride);
        cpa!(scope, index_2 += x2_stride);
        cpa!(scope, inp_2 = input[index_2]);
        cpa!(scope, index_3 = index_base);
        cpa!(scope, index_3 += y0_stride);
        cpa!(scope, index_3 += x3_stride);
        cpa!(scope, inp_3 = input[index_3]);

        let coefficients0 = Self::cubic_interp1d(scope, inp_0, inp_1, inp_2, inp_3, xw);

        cpa!(scope, index_0 = index_base);
        cpa!(scope, index_0 += y1_stride);
        cpa!(scope, index_0 += x0_stride);
        cpa!(scope, inp_0 = input[index_0]);
        cpa!(scope, index_1 = index_base);
        cpa!(scope, index_1 += y1_stride);
        cpa!(scope, index_1 += x1_stride);
        cpa!(scope, inp_1 = input[index_1]);
        cpa!(scope, index_2 = index_base);
        cpa!(scope, index_2 += y1_stride);
        cpa!(scope, index_2 += x2_stride);
        cpa!(scope, inp_2 = input[index_2]);
        cpa!(scope, index_3 = index_base);
        cpa!(scope, index_3 += y1_stride);
        cpa!(scope, index_3 += x3_stride);
        cpa!(scope, inp_3 = input[index_3]);

        let coefficients1 = Self::cubic_interp1d(scope, inp_0, inp_1, inp_2, inp_3, xw);

        cpa!(scope, index_0 = index_base);
        cpa!(scope, index_0 += y2_stride);
        cpa!(scope, index_0 += x0_stride);
        cpa!(scope, inp_0 = input[index_0]);
        cpa!(scope, index_1 = index_base);
        cpa!(scope, index_1 += y2_stride);
        cpa!(scope, index_1 += x1_stride);
        cpa!(scope, inp_1 = input[index_1]);
        cpa!(scope, index_2 = index_base);
        cpa!(scope, index_2 += y2_stride);
        cpa!(scope, index_2 += x2_stride);
        cpa!(scope, inp_2 = input[index_2]);
        cpa!(scope, index_3 = index_base);
        cpa!(scope, index_3 += y2_stride);
        cpa!(scope, index_3 += x3_stride);
        cpa!(scope, inp_3 = input[index_3]);

        let coefficients2 = Self::cubic_interp1d(scope, inp_0, inp_1, inp_2, inp_3, xw);

        cpa!(scope, index_0 = index_base);
        cpa!(scope, index_0 += y3_stride);
        cpa!(scope, index_0 += x0_stride);
        cpa!(scope, inp_0 = input[index_0]);
        cpa!(scope, index_1 = index_base);
        cpa!(scope, index_1 += y3_stride);
        cpa!(scope, index_1 += x1_stride);
        cpa!(scope, inp_1 = input[index_1]);
        cpa!(scope, index_2 = index_base);
        cpa!(scope, index_2 += y3_stride);
        cpa!(scope, index_2 += x2_stride);
        cpa!(scope, inp_2 = input[index_2]);
        cpa!(scope, index_3 = index_base);
        cpa!(scope, index_3 += y3_stride);
        cpa!(scope, index_3 += x3_stride);
        cpa!(scope, inp_3 = input[index_3]);

        let coefficients3 = Self::cubic_interp1d(scope, inp_0, inp_1, inp_2, inp_3, xw);

        let val = Self::cubic_interp1d(
            scope,
            coefficients0,
            coefficients1,
            coefficients2,
            coefficients3,
            yw,
        );

        cpa!(scope, output[id] = val);
    }

    fn min(scope: &mut Scope, a: Variable, b: Variable) -> Variable {
        let cond = scope.create_local(Elem::Bool);
        let res = scope.create_local(a.item());

        cpa!(scope, cond = a < b);
        cpa!(scope, if(cond).then(|scope|{
            cpa!(scope, res = a);
        }).else(|scope|{
            cpa!(scope, res = b);
        }));

        res
    }

    fn cubic_interp1d(
        scope: &mut Scope,
        x0: Variable,
        x1: Variable,
        x2: Variable,
        x3: Variable,
        t: Variable,
    ) -> Variable {
        let item = x0.item();
        let x = scope.create_local(item);
        let a: Variable = scope.create_with_value(-0.75, item);
        let one: Variable = scope.create_with_value(1, item);
        let two: Variable = scope.create_with_value(2, item);
        let cubic = scope.create_local(item);
        let cubic_tmp = scope.create_local(item);

        cpa!(scope, x = t + one);
        let coeffs0 = Self::cubic_convolution2(scope, x, a);

        let coeffs1 = Self::cubic_convolution1(scope, t, a);

        cpa!(scope, x = one - t);
        let coeffs2 = Self::cubic_convolution1(scope, x, a);

        cpa!(scope, x = two - t);
        let coeffs3 = Self::cubic_convolution2(scope, x, a);

        cpa!(scope, cubic = x0 * coeffs0);
        cpa!(scope, cubic_tmp = x1 * coeffs1);
        cpa!(scope, cubic += cubic_tmp);
        cpa!(scope, cubic_tmp = x2 * coeffs2);
        cpa!(scope, cubic += cubic_tmp);
        cpa!(scope, cubic_tmp = x3 * coeffs3);
        cpa!(scope, cubic += cubic_tmp);

        cubic
    }

    fn cubic_convolution1(scope: &mut Scope, x: Variable, a: Variable) -> Variable {
        let item = x.item();
        let conv = scope.create_local(item);
        let tmp = scope.create_local(item);
        let one = scope.create_with_value(1, item);
        let two = scope.create_with_value(2, item);
        let three = scope.create_with_value(3, item);

        cpa!(scope, conv = a + two);
        cpa!(scope, conv *= x);
        cpa!(scope, tmp = a + three);
        cpa!(scope, conv = conv - tmp);
        cpa!(scope, conv *= x);
        cpa!(scope, conv *= x);
        cpa!(scope, conv += one);

        conv
    }

    fn cubic_convolution2(scope: &mut Scope, x: Variable, a: Variable) -> Variable {
        let item = x.item();
        let conv = scope.create_local(item);
        let tmp = scope.create_local(item);
        let four = scope.create_with_value(4, item);
        let five = scope.create_with_value(5, item);
        let eight = scope.create_with_value(8, item);

        cpa!(scope, conv = a * x);
        cpa!(scope, tmp = five * a);
        cpa!(scope, conv = conv - tmp);
        cpa!(scope, conv *= x);
        cpa!(scope, tmp = eight * a);
        cpa!(scope, conv += tmp);
        cpa!(scope, conv *= x);
        cpa!(scope, tmp = four * a);
        cpa!(scope, conv = conv - tmp);

        conv
    }
}

impl<R: JitRuntime, E: JitElement> Kernel for InterpolateBicubicEagerKernel<R, E> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item = E::cube_elem().into();

        let input = Variable::GlobalInputArray { id: 0, item };
        let output = Variable::GlobalOutputArray { id: 0, item };

        InterpolateBicubicShader {
            input,
            output,
            _elem: PhantomData::<E>,
        }
        .expand(&mut scope);

        scope.write_global_custom(output);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };

        let out = OutputInfo::Array { item };

        let info = KernelExpansion {
            inputs: vec![input],
            outputs: vec![out],
            scope,
        };

        let settings = KernelSettings::default();
        KernelIntegrator::new(info).integrate(settings)
    }

    fn id(&self) -> cubecl::KernelId {
        cubecl::KernelId::new::<Self>()
    }
}

pub(crate) fn interpolate_bicubic_launch<R: JitRuntime, E: JitElement>(
    input: JitTensor<R, E>,
    output: JitTensor<R, E>,
) -> JitTensor<R, E> {
    let kernel = InterpolateBicubicEagerKernel::<R, E>::new();

    Execution::start(kernel, input.client.clone())
        .inputs(&[input.as_handle_ref()])
        .outputs(&[output.as_handle_ref()])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}
