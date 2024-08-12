use crate::{
    element::JitElement, kernel::Kernel, ops::numeric::empty_device, tensor::JitTensor, JitRuntime,
};
use cubecl::prelude::*;
use cubecl::{
    calculate_cube_count_elemwise, frontend::TensorHandleRef, CubeCountSettings, CubeDim, Execution,
};

#[cube(launch_unchecked)]
fn select_kernel<T: Numeric>(
    input: &Tensor<T>,
    indices: &Tensor<I32>,
    output: &mut Tensor<T>,
    dim: &UInt,
) {
    let id = ABSOLUTE_POS;
    let mut offset_input = UInt::new(0);
    let rank = output.rank();
    for i in range(UInt::new(0), rank, Comptime::new(false)) {
        let stride_input = input.stride(i);
        let stride_output = output.stride(i);
        let shape_output = output.shape(i);
        let mut offset_local = id / stride_output;
        offset_local = offset_local % shape_output;

        if i == *dim {
            offset_local = UInt::cast_from(indices[offset_local]);
            offset_local *= stride_input;
        } else {
            offset_local *= stride_input;
        }
        offset_input += offset_local;
    }
    let value = input[offset_input];
    output[id] = value;
}
//pub fn expand(self, scope: &mut Scope) {
//    let input = self.input;
//    let indices = self.indices;
//    let output = self.output;
//    let id = Variable::AbsolutePos;
//    let offset_input = scope.zero(Elem::UInt);
//
//    cpa!(
//        scope,
//        range(0u32, Variable::Rank).for_each(|i, scope| {
//            let stride_input = scope.create_local(Elem::UInt);
//            let stride_output = scope.create_local(Elem::UInt);
//            let shape_output = scope.create_local(Elem::UInt);
//
//            cpa!(scope, stride_input = stride(input, i));
//            cpa!(scope, stride_output = stride(output, i));
//            cpa!(scope, shape_output = shape(output, i));
//
//            let offset_local = scope.create_local(Elem::UInt);
//            cpa!(scope, offset_local = id / stride_output);
//            cpa!(scope, offset_local = offset_local % shape_output);
//
//            let dim_index = scope.create_local(Elem::Bool);
//            cpa!(scope, dim_index = i == self.dim);
//
//            cpa!(scope, if(dim_index).then(|scope| {
//                cpa!(scope, offset_local = indices[offset_local]);
//                cpa!(scope, offset_local = offset_local * stride_input);
//            }).else(|scope| {
//                cpa!(scope, offset_local = offset_local * stride_input);
//            }));
//
//            cpa!(scope, offset_input += offset_local);
//        })
//    );
//
//    let value = scope.create_local(input.item());
//    cpa!(scope, value = input[offset_input]);
//    cpa!(scope, output[id] = value);
//}

pub(crate) fn select<R: JitRuntime, E: JitElement, I: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    dim: usize,
    indices: JitTensor<R, I, 1>,
) -> JitTensor<R, E, D> {
    let mut shape_output = tensor.shape.clone();
    shape_output.dims[dim] = indices.shape.dims[0];
    let num_elems = indices.shape.dims[0];
    let mut total_elem = 1;
    for dim_size in shape_output.dims.iter() {
        total_elem *= dim_size
    }
    let mut shapes = [1; D];
    let mut strides = [num_elems; D];
    shapes[D - 1] = num_elems;
    strides[D - 1] = 1;

    let output = empty_device(tensor.client.clone(), tensor.device.clone(), shape_output);

    let cube_dim = CubeDim::default();

    let cube_count = calculate_cube_count_elemwise(total_elem, cube_dim);

    unsafe {
        select_kernel::launch_unchecked::<E::Primitive, R>(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            TensorArg::from_raw_parts(&indices.handle, &strides, &shapes, 1),
            output.as_tensor_arg(1),
            ScalarArg::new(dim as u32),
        )
    };

    output
}
