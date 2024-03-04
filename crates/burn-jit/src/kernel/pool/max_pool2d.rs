use burn_tensor::{ops::conv::calculate_pool_output_size, ElementConversion, Shape};
use std::marker::PhantomData;

use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Item, Scope, Variable, Visibility},
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler, EagerHandle,
        InputInfo, OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    kernel::{DynamicKernelSource, SourceTemplate},
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime, RuntimeInt,
};

#[derive(new)]
struct MaxPool2dEagerKernel<R: Runtime, E: JitElement> {
    kernel_size: [usize; 2],
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

#[derive(new)]
struct MaxPool2dWithIndicesEagerKernel<R: Runtime, E: JitElement> {
    kernel_size: [usize; 2],
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct MaxPool2dComputeShader {
    x: Variable,
    output: Variable,
    kernel_size: [usize; 2],
    indices: Option<Variable>,
}

impl MaxPool2dComputeShader {
    fn expand(self, scope: &mut Scope) {
        let x = self.x;
        let output = self.output;
        let id = Variable::Id;

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

        gpu!(scope, input_stride_0 = stride(x, 0u32));
        gpu!(scope, input_stride_1 = stride(x, 1u32));
        gpu!(scope, input_stride_2 = stride(x, 2u32));
        gpu!(scope, input_stride_3 = stride(x, 3u32));

        gpu!(scope, input_shape_2 = shape(x, 2u32));
        gpu!(scope, input_shape_3 = shape(x, 3u32));

        gpu!(scope, output_stride_0 = stride(output, 0u32));
        gpu!(scope, output_stride_1 = stride(output, 1u32));
        gpu!(scope, output_stride_2 = stride(output, 2u32));
        gpu!(scope, output_stride_3 = stride(output, 3u32));

        gpu!(scope, output_shape_0 = shape(output, 0u32));
        gpu!(scope, output_shape_1 = shape(output, 1u32));
        gpu!(scope, output_shape_2 = shape(output, 2u32));
        gpu!(scope, output_shape_3 = shape(output, 3u32));

        let pool_stride_0 = Variable::GlobalScalar(0, Elem::UInt);
        let pool_stride_1 = Variable::GlobalScalar(1, Elem::UInt);
        let dilation_0 = Variable::GlobalScalar(2, Elem::UInt);
        let dilation_1 = Variable::GlobalScalar(3, Elem::UInt);
        let padding_0 = Variable::GlobalScalar(4, Elem::UInt);
        let padding_1 = Variable::GlobalScalar(5, Elem::UInt);

        let [kernel_size_0, kernel_size_1] = self.kernel_size;

        let b = scope.create_local(Elem::UInt);
        let c = scope.create_local(Elem::UInt);
        let oh = scope.create_local(Elem::UInt);
        let ow = scope.create_local(Elem::UInt);

        gpu!(scope, b = id / output_stride_0);
        gpu!(scope, b = b % output_shape_0);

        gpu!(scope, c = id / output_stride_1);
        gpu!(scope, c = c % output_shape_1);

        gpu!(scope, oh = id / output_stride_2);
        gpu!(scope, oh = oh % output_shape_2);

        gpu!(scope, ow = id / output_stride_3);
        gpu!(scope, ow = ow % output_shape_3);

        let tmp = scope.create_local(Elem::UInt);
        let ih = scope.create_local(Elem::UInt);
        let iw = scope.create_local(Elem::UInt);

        let ih_pad = scope.create_local(Elem::UInt);
        let iw_pad = scope.create_local(Elem::UInt);
        let result = scope.create_local(x.item());

        let cond = scope.create_local(Elem::Bool);
        let cond_tmp = scope.create_local(Elem::Bool);

        let index_input = scope.create_local(Elem::UInt);
        let index_input_1 = scope.create_local(Elem::UInt);
        let index_input_2 = scope.create_local(Elem::UInt);
        let index_input_3 = scope.create_local(Elem::UInt);
        let index_input_4 = scope.create_local(Elem::UInt);

        let is_max = scope.create_local(Elem::Bool);
        let max_val = scope.create_local(x.item());
        let max_index = self.indices.map(|_| scope.create_local(Elem::UInt));
        gpu!(scope, max_val = cast(-32767.0));

        (0..kernel_size_0).for_each(|kh| {
            gpu!(scope, ih = oh * pool_stride_0);
            gpu!(scope, tmp = kh * dilation_0);
            gpu!(scope, ih += tmp);

            // Up
            gpu!(scope, cond = ih < padding_0);
            // Down
            gpu!(scope, tmp = input_shape_2 + padding_0);
            gpu!(scope, cond_tmp = ih >= tmp);
            gpu!(scope, cond = cond || cond_tmp);
            gpu!(scope, cond = !cond);

            gpu!(scope, if (cond).then(|scope| {
                (0..kernel_size_1).for_each(|kw| {
                    gpu!(scope, iw = ow * pool_stride_1);
                    gpu!(scope, tmp = kw * dilation_1);
                    gpu!(scope, iw = iw + tmp);

                    // Left
                    gpu!(scope, cond = iw < padding_1);
                    // Right
                    gpu!(scope, tmp = input_shape_3 + padding_1);
                    gpu!(scope, cond_tmp = iw >= tmp);
                    gpu!(scope, cond = cond || cond_tmp);
                    gpu!(scope, cond = !cond);

                    gpu!(scope, if (cond).then(|scope| {
                        gpu!(scope, ih_pad = ih - padding_0);
                        gpu!(scope, iw_pad = iw - padding_1);

                        gpu!(scope, index_input_1 = b * input_stride_0);
                        gpu!(scope, index_input_2 = c * input_stride_1);
                        gpu!(scope, index_input_3 = ih_pad * input_stride_2);
                        gpu!(scope, index_input_4 = iw_pad * input_stride_3);

                        gpu!(scope, index_input = index_input_1);
                        gpu!(scope, index_input += index_input_2);
                        gpu!(scope, index_input += index_input_3);
                        gpu!(scope, index_input += index_input_4);

                        gpu!(scope, result = x[index_input]);

                        gpu!(scope, is_max = result > max_val);

                        gpu!(scope, if(is_max).then(|scope|{
                            gpu!(scope, max_val = result);
                            if let Some(max_index) = max_index {
                                gpu!(scope, max_index = ih_pad * input_shape_2);
                                gpu!(scope, max_index += iw_pad);
                            }
                        }));
                    }));
                });
            }));
        });

        gpu!(scope, output[id] = max_val);

        if let Some(indices) = self.indices {
            let max_index = max_index.unwrap();
            gpu!(scope, indices[id] = max_index);
        }
    }
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for MaxPool2dEagerKernel<R, E> {
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let x = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

        scope.write_global_custom(output);

        MaxPool2dComputeShader {
            x,
            output,
            kernel_size: self.kernel_size,
            indices: None,
        }
        .expand(&mut scope);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let scalars = InputInfo::Scalar {
            elem: Elem::UInt,
            size: 6,
        };
        let output = OutputInfo::Array { item };

        let info = CompilationInfo {
            inputs: vec![input, scalars],
            outputs: vec![output],
            scope,
        };

        let settings = CompilationSettings::default();
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!(
            "{:?}k={:?}",
            core::any::TypeId::of::<Self>(),
            self.kernel_size,
        )
    }
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for MaxPool2dWithIndicesEagerKernel<R, E> {
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let x = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);
        let indices = Variable::GlobalOutputArray(1, Item::Scalar(Elem::Int));

        scope.write_global_custom(output);

        MaxPool2dComputeShader {
            x,
            output,
            kernel_size: self.kernel_size,
            indices: Some(indices),
        }
        .expand(&mut scope);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let scalars = InputInfo::Scalar {
            elem: Elem::UInt,
            size: 6,
        };
        let output = OutputInfo::Array { item };
        let indices = OutputInfo::Array {
            item: Item::Scalar(Elem::Int),
        };

        let info = CompilationInfo {
            inputs: vec![input, scalars],
            outputs: vec![output, indices],
            scope,
        };

        let settings = CompilationSettings::default();
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!(
            "{:?}k={:?}",
            core::any::TypeId::of::<Self>(),
            self.kernel_size,
        )
    }
}

pub(crate) fn max_pool2d_with_indices<R: Runtime, E: JitElement, I: JitElement>(
    x: JitTensor<R, E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> (JitTensor<R, E, 4>, JitTensor<R, I, 4>) {
    let [batch_size, channels, _, _] = x.shape.dims;

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        x.shape.dims[2],
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        x.shape.dims[3],
    );

    let shape_out = Shape::new([batch_size, channels, size_0, size_1]);
    let output = empty_device(x.client.clone(), x.device.clone(), shape_out.clone());
    let indices = empty_device(x.client.clone(), x.device.clone(), shape_out);

    let kernel = MaxPool2dWithIndicesEagerKernel::new(kernel_size);
    execute_dynamic::<R, MaxPool2dWithIndicesEagerKernel<R, E>, I>(
        &[EagerHandle::new(&x.handle, &x.strides, &x.shape.dims)],
        &[
            EagerHandle::new(&output.handle, &output.strides, &output.shape.dims),
            EagerHandle::new(&indices.handle, &indices.strides, &indices.shape.dims),
        ],
        Some(&[
            (stride[0] as i32).elem(),
            (stride[1] as i32).elem(),
            (dilation[0] as i32).elem(),
            (dilation[1] as i32).elem(),
            (padding[0] as i32).elem(),
            (padding[1] as i32).elem(),
        ]),
        kernel,
        WorkgroupLaunch::Output { pos: 0 },
        x.client,
    );

    (output, indices)
}

pub(crate) fn max_pool2d<R: Runtime, E: JitElement>(
    x: JitTensor<R, E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> JitTensor<R, E, 4> {
    let [batch_size, channels, _, _] = x.shape.dims;

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        x.shape.dims[2],
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        x.shape.dims[3],
    );

    let shape_out = Shape::new([batch_size, channels, size_0, size_1]);
    let output = empty_device(x.client.clone(), x.device.clone(), shape_out);

    let kernel = MaxPool2dEagerKernel::new(kernel_size);

    execute_dynamic::<R, MaxPool2dEagerKernel<R, E>, RuntimeInt<R>>(
        &[EagerHandle::new(&x.handle, &x.strides, &x.shape.dims)],
        &[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )],
        Some(&[
            (stride[0] as i32).elem(),
            (stride[1] as i32).elem(),
            (dilation[0] as i32).elem(),
            (dilation[1] as i32).elem(),
            (padding[0] as i32).elem(),
            (padding[1] as i32).elem(),
        ]),
        kernel,
        WorkgroupLaunch::Output { pos: 0 },
        x.client,
    );

    output
}

// #[cfg(test)]
// mod tests {
//     use crate::tests::{ReferenceBackend, TestBackend};
//     use burn_tensor::{module, Distribution, Tensor};
//
//     #[test]
//     pub fn max_pool2d_should_work_with_multiple_invocations() {
//         let tensor = Tensor::<TestBackend, 4>::random(
//             [32, 32, 32, 32],
//             Distribution::Default,
//             &Default::default(),
//         );
//         let tensor_ref =
//             Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data(), &Default::default());
//         let kernel_size = [3, 3];
//         let stride = [2, 2];
//         let padding = [1, 1];
//         let dilation = [1, 1];
//
//         let pooled = module::max_pool2d(tensor, kernel_size, stride, padding, dilation);
//         let pooled_ref = module::max_pool2d(tensor_ref, kernel_size, stride, padding, dilation);
//
//         pooled
//             .into_data()
//             .assert_approx_eq(&pooled_ref.into_data(), 3);
//     }
//
//     #[test]
//     pub fn max_pool2d_with_indices_should_work_with_multiple_invocations() {
//         let tensor = Tensor::<TestBackend, 4>::random(
//             [32, 32, 32, 32],
//             Distribution::Default,
//             &Default::default(),
//         );
//         let tensor_ref =
//             Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data(), &Default::default());
//         let kernel_size = [3, 3];
//         let stride = [2, 2];
//         let padding = [1, 1];
//         let dilation = [1, 1];
//
//         let (pooled, indices) =
//             module::max_pool2d_with_indices(tensor, kernel_size, stride, padding, dilation);
//         let (pooled_ref, indices_ref) =
//             module::max_pool2d_with_indices(tensor_ref, kernel_size, stride, padding, dilation);
//
//         pooled
//             .into_data()
//             .assert_approx_eq(&pooled_ref.into_data(), 3);
//         assert_eq!(indices.into_data(), indices_ref.into_data().convert());
//     }
// }
