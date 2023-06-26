use crate::{
    comparison, comparison_elem, comparison_elem_inplace, comparison_inplace,
    context::WorkGroup,
    element::WgpuElement,
    kernel::{
        build_info, cat, comparison, comparison_elem, comparison_elem_inplace, comparison_inplace,
        mask_fill, mask_fill_inplace, mask_where, mask_where_inplace, KernelSettings,
    },
    kernel_wgsl,
    pool::get_context,
    tensor::WgpuTensor,
    GraphicsApi, WgpuDevice,
};
use burn_tensor::{backend::Backend, Data, Shape};
use std::{marker::PhantomData, mem, ops::Range};

pub type FloatElem<B> = <B as Backend>::FloatElem;
pub type Device<B> = <B as Backend>::Device;

pub type FloatTensor<B, const D: usize> = <B as Backend>::TensorPrimitive<D>;

pub type IntElem<B> = <B as Backend>::IntElem;
pub type IntTensor<B, const D: usize> = <B as Backend>::IntTensorPrimitive<D>;
pub type BoolTensor<B, const D: usize> = <B as Backend>::BoolTensorPrimitive<D>;

pub struct BaseOps<G: GraphicsApi> {
    _g: PhantomData<G>,
}

comparison!(Equal, "==");
comparison!(Greater, ">");
comparison!(GreaterEqual, ">=");
comparison!(Lower, "<");
comparison!(LowerEqual, "<=");

comparison_inplace!(EqualInplace, "==");
comparison_inplace!(GreaterInplace, ">");
comparison_inplace!(GreaterEqualInplace, ">=");
comparison_inplace!(LowerInplace, "<");
comparison_inplace!(LowerEqualInplace, "<=");

comparison_elem!(EqualElem, "==");
comparison_elem!(GreaterElem, ">");
comparison_elem!(GreaterEqualElem, ">=");
comparison_elem!(LowerElem, "<");
comparison_elem!(LowerEqualElem, "<=");

comparison_elem_inplace!(EqualElemInplace, "==");
comparison_elem_inplace!(GreaterElemInplace, ">");
comparison_elem_inplace!(GreaterEqualElemInplace, ">=");
comparison_elem_inplace!(LowerElemInplace, "<");
comparison_elem_inplace!(LowerEqualElemInplace, "<=");

impl<G: GraphicsApi> BaseOps<G> {
    pub fn from_data<E: WgpuElement, const D: usize>(
        data: Data<E, D>,
        device: &WgpuDevice,
    ) -> WgpuTensor<E, D> {
        let context = get_context::<G>(device);
        let buffer = context.create_buffer_with_data_options(E::as_bytes(&data.value), true);

        WgpuTensor::new(context, data.shape, buffer)
    }

    pub fn into_data<E: WgpuElement, const D: usize>(tensor: WgpuTensor<E, D>) -> Data<E, D> {
        let tensor = Self::into_continuous(tensor);
        let bytes = tensor.context.read_buffer(tensor.buffer);
        let values = E::from_bytes(&bytes);

        Data::new(values.to_vec(), tensor.shape)
    }

    pub fn to_device<E: WgpuElement, const D: usize>(
        tensor: WgpuTensor<E, D>,
        device: &WgpuDevice,
    ) -> WgpuTensor<E, D> {
        if &tensor.context.device == device {
            return tensor;
        }

        let context = get_context::<G>(device);
        tensor.to_context(context)
    }

    pub fn empty<E: WgpuElement, const D: usize>(
        shape: Shape<D>,
        device: &WgpuDevice,
    ) -> WgpuTensor<E, D> {
        let context = get_context::<G>(device);
        let buffer = context.create_buffer(shape.num_elements() * core::mem::size_of::<E>());

        WgpuTensor::new(context, shape, buffer)
    }

    pub fn swap_dims<E: WgpuElement, const D: usize>(
        mut tensor: WgpuTensor<E, D>,
        dim1: usize,
        dim2: usize,
    ) -> WgpuTensor<E, D> {
        tensor.strides.swap(dim1, dim2);

        tensor.shape.dims.swap(dim1, dim2);

        tensor
    }

    pub fn reshape<E: WgpuElement, const D1: usize, const D2: usize>(
        tensor: WgpuTensor<E, D1>,
        shape: Shape<D2>,
    ) -> WgpuTensor<E, D2> {
        // TODO: Not force standard layout all the time (improve performance).
        let tensor = Self::into_continuous(tensor);

        WgpuTensor::new(tensor.context, shape, tensor.buffer)
    }

    pub fn into_continuous<E: WgpuElement, const D: usize>(
        tensor: WgpuTensor<E, D>,
    ) -> WgpuTensor<E, D> {
        if tensor.is_continuous() {
            return tensor;
        }

        kernel_wgsl!(ContinuousRaw, "../template/continuous.wgsl");

        let buffer = tensor
            .context
            .create_buffer(tensor.shape.num_elements() * core::mem::size_of::<E>());
        let output = WgpuTensor::new(tensor.context.clone(), tensor.shape.clone(), buffer);
        let info = build_info(&[&tensor, &output]);
        let info_buffer = tensor
            .context
            .create_buffer_with_data(bytemuck::cast_slice(&info));

        let kernel = tensor
            .context
            .compile_static::<KernelSettings<ContinuousRaw, E, i32, 256, 1, 1>>();

        tensor.context.execute(
            WorkGroup::new(
                f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
                1,
                1,
            ),
            kernel,
            &[&tensor.buffer, &output.buffer, &info_buffer],
        );

        output
    }

    pub fn index<E: WgpuElement, const D1: usize, const D2: usize>(
        tensor: WgpuTensor<E, D1>,
        indexes: [Range<usize>; D2],
    ) -> WgpuTensor<E, D1> {
        kernel_wgsl!(IndexRaw, "../template/index/index.wgsl");

        let mut dims = tensor.shape.dims;

        for i in 0..D2 {
            dims[i] = indexes[i].end - indexes[i].start;
        }

        let shape_output = Shape::new(dims);

        let buffer = tensor
            .context
            .create_buffer(shape_output.num_elements() * core::mem::size_of::<E>());
        let output = WgpuTensor::new(tensor.context.clone(), shape_output, buffer);
        let mut info = build_info(&[&tensor, &output]);

        for i in 0..D1 {
            let start = indexes.get(i).map(|index| index.start).unwrap_or(0);
            info.push(start as u32);
        }

        let info_buffer = tensor
            .context
            .create_buffer_with_data(bytemuck::cast_slice(&info));

        let kernel = tensor
            .context
            .compile_static::<KernelSettings<IndexRaw, E, i32, 256, 1, 1>>();

        tensor.context.execute(
            WorkGroup::new(
                f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
                1,
                1,
            ),
            kernel,
            &[&tensor.buffer, &output.buffer, &info_buffer],
        );

        output
    }

    pub fn index_assign<E: WgpuElement, const D1: usize, const D2: usize>(
        tensor: WgpuTensor<E, D1>,
        indexes: [Range<usize>; D2],
        value: WgpuTensor<E, D1>,
    ) -> WgpuTensor<E, D1> {
        kernel_wgsl!(
            IndexAssignInplaceRaw,
            "../template/index/index_assign_inplace.wgsl"
        );

        let tensor = match tensor.can_mut() {
            true => tensor,
            false => tensor.copy(),
        };

        let mut info = build_info(&[&tensor, &value]);

        for i in 0..D1 {
            let start = indexes.get(i).map(|index| index.start).unwrap_or(0);
            info.push(start as u32);
        }

        let info_buffer = tensor
            .context
            .create_buffer_with_data(bytemuck::cast_slice(&info));

        let kernel = tensor
            .context
            .compile_static::<KernelSettings<IndexAssignInplaceRaw, E, i32, 256, 1, 1>>();

        tensor.context.execute(
            WorkGroup::new(
                f32::ceil(value.shape.num_elements() as f32 / 256_f32) as u32,
                1,
                1,
            ),
            kernel,
            &[&tensor.buffer, &value.buffer, &info_buffer],
        );

        tensor
    }

    pub fn index_select<E: WgpuElement, I: WgpuElement, const D: usize>(
        tensor: WgpuTensor<E, D>,
        dim: usize,
        indexes: WgpuTensor<I, 1>,
    ) -> WgpuTensor<E, D> {
        kernel_wgsl!(IndexSelect, "../template/index/index_select.wgsl");

        let mut output_shape = tensor.shape.clone();
        output_shape.dims[dim] = indexes.shape.dims[0];

        let buffer = tensor
            .context
            .create_buffer(std::mem::size_of::<E>() * output_shape.num_elements());
        let output = WgpuTensor::new(tensor.context.clone(), output_shape, buffer);

        let mut info = build_info(&[&tensor, &output]);
        info.push(dim as u32);

        let info_buffer = tensor
            .context
            .create_buffer_with_data(bytemuck::cast_slice(&info));

        let kernel = tensor
            .context
            .compile_static::<KernelSettings<IndexSelect, E, I, 256, 1, 1>>();

        tensor.context.execute(
            WorkGroup::new(
                f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
                1,
                1,
            ),
            kernel,
            &[
                &tensor.buffer,
                &indexes.buffer,
                &output.buffer,
                &info_buffer,
            ],
        );

        output
    }

    pub fn index_select_assign<E: WgpuElement, I: WgpuElement, const D: usize, const D2: usize>(
        tensor: WgpuTensor<E, D>,
        dim: usize,
        indexes: WgpuTensor<I, 1>,
        values: WgpuTensor<E, D2>,
    ) -> WgpuTensor<E, D> {
        kernel_wgsl!(
            IndexSelectAssignInplace,
            "../template/index/index_select_assign_inplace.wgsl"
        );

        let tensor = match tensor.can_mut() {
            true => tensor,
            false => tensor.copy(),
        };

        let mut shape = tensor.shape.clone();
        shape.dims[dim] = values.shape.dims[dim];
        let values = WgpuTensor::new(values.context, shape, values.buffer);
        let mut info = build_info(&[&tensor, &values]);
        info.push(dim as u32);

        let info_buffer = tensor
            .context
            .create_buffer_with_data(bytemuck::cast_slice(&info));

        let kernel = tensor
            .context
            .compile_static::<KernelSettings<IndexSelectAssignInplace, E, I, 256, 1, 1>>();

        let mut shape_tmp = values.shape;
        shape_tmp.dims[dim] = 1; // Just one thread for the dim.

        tensor.context.execute(
            WorkGroup::new(
                f32::ceil(shape_tmp.num_elements() as f32 / 256_f32) as u32,
                1,
                1,
            ),
            kernel,
            &[
                &tensor.buffer,
                &indexes.buffer,
                &values.buffer,
                &info_buffer,
            ],
        );

        tensor
    }

    pub fn equal<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<u32, D> {
        let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

        if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
            return comparison_inplace::<EqualInplace, E, D>(lhs, rhs);
        }
        if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
            return comparison_inplace::<EqualInplace, E, D>(rhs, lhs);
        }

        comparison::<Equal, E, D>(lhs, rhs)
    }

    pub fn greater<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<u32, D> {
        let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

        if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
            return comparison_inplace::<GreaterInplace, E, D>(lhs, rhs);
        }
        if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
            return comparison_inplace::<LowerInplace, E, D>(rhs, lhs);
        }

        comparison::<Greater, E, D>(lhs, rhs)
    }

    pub fn greater_equal<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<u32, D> {
        let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

        if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
            return comparison_inplace::<GreaterEqualInplace, E, D>(lhs, rhs);
        }
        if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
            return comparison_inplace::<LowerEqualInplace, E, D>(rhs, lhs);
        }

        comparison::<GreaterEqual, E, D>(lhs, rhs)
    }

    pub fn lower<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<u32, D> {
        let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

        if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
            return comparison_inplace::<LowerInplace, E, D>(lhs, rhs);
        }
        if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
            return comparison_inplace::<GreaterInplace, E, D>(rhs, lhs);
        }

        comparison::<Lower, E, D>(lhs, rhs)
    }

    pub fn lower_equal<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<u32, D> {
        let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

        if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
            return comparison_inplace::<LowerEqualInplace, E, D>(lhs, rhs);
        }
        if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
            return comparison_inplace::<GreaterEqualInplace, E, D>(rhs, lhs);
        }

        comparison::<LowerEqual, E, D>(lhs, rhs)
    }

    pub fn equal_elem<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<u32, D> {
        if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
            return comparison_elem_inplace::<EqualElemInplace, E, D>(lhs, rhs);
        }

        comparison_elem::<EqualElem, E, D>(lhs, rhs)
    }

    pub fn greater_elem<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<u32, D> {
        if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
            return comparison_elem_inplace::<GreaterElemInplace, E, D>(lhs, rhs);
        }

        comparison_elem::<GreaterElem, E, D>(lhs, rhs)
    }

    pub fn lower_elem<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<u32, D> {
        if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
            return comparison_elem_inplace::<LowerElemInplace, E, D>(lhs, rhs);
        }

        comparison_elem::<LowerElem, E, D>(lhs, rhs)
    }

    pub fn greater_equal_elem<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<u32, D> {
        if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
            return comparison_elem_inplace::<GreaterEqualElemInplace, E, D>(lhs, rhs);
        }

        comparison_elem::<GreaterEqualElem, E, D>(lhs, rhs)
    }

    pub fn lower_equal_elem<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<u32, D> {
        if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
            return comparison_elem_inplace::<LowerEqualElemInplace, E, D>(lhs, rhs);
        }

        comparison_elem::<LowerEqualElem, E, D>(lhs, rhs)
    }

    pub fn mask_fill<E: WgpuElement, const D: usize>(
        tensor: WgpuTensor<E, D>,
        mask: WgpuTensor<u32, D>,
        value: E,
    ) -> WgpuTensor<E, D> {
        if tensor.can_mut() {
            return mask_fill_inplace(tensor, mask, value);
        }

        mask_fill(tensor, mask, value)
    }

    pub fn mask_where<E: WgpuElement, const D: usize>(
        tensor: WgpuTensor<E, D>,
        mask: WgpuTensor<u32, D>,
        value: WgpuTensor<E, D>,
    ) -> WgpuTensor<E, D> {
        if tensor.can_mut_broadcast(&value) {
            return mask_where_inplace(tensor, mask, value, 1);
        }
        if value.can_mut_broadcast(&tensor) {
            return mask_where_inplace(value, mask, tensor, 0);
        }

        mask_where(tensor, mask, value)
    }

    pub fn cat<E: WgpuElement, const D: usize>(
        tensors: Vec<WgpuTensor<E, D>>,
        dim: usize,
    ) -> WgpuTensor<E, D> {
        cat(tensors, dim)
    }

    pub fn gather<E: WgpuElement, I: WgpuElement, const D: usize>(
        dim: usize,
        tensor: WgpuTensor<E, D>,
        indexes: WgpuTensor<I, D>,
    ) -> WgpuTensor<E, D> {
        kernel_wgsl!(Gather, "../template/gather.wgsl");
        let shape_output = indexes.shape.clone();
        let indexes = Self::into_continuous(indexes);

        let buffer = tensor
            .context
            .create_buffer(shape_output.num_elements() * core::mem::size_of::<E>());
        let output = WgpuTensor::new(tensor.context.clone(), shape_output, buffer);
        let mut info = build_info(&[&tensor, &output]);
        info.push(dim as u32);
        let info_buffer = tensor
            .context
            .create_buffer_with_data(bytemuck::cast_slice(&info));

        let kernel = tensor
            .context
            .compile_static::<KernelSettings<Gather, E, i32, 256, 1, 1>>();

        tensor.context.execute(
            WorkGroup::new(
                f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
                1,
                1,
            ),
            kernel,
            &[
                &tensor.buffer,
                &indexes.buffer,
                &output.buffer,
                &info_buffer,
            ],
        );

        output
    }

    pub fn scatter<E: WgpuElement, I: WgpuElement, const D: usize>(
        dim: usize,
        tensor: WgpuTensor<E, D>,
        indexes: WgpuTensor<I, D>,
        value: WgpuTensor<E, D>,
    ) -> WgpuTensor<E, D> {
        kernel_wgsl!(Scatter, "../template/scatter.wgsl");

        const WORKGROUP: usize = 256;

        let indexes = Self::into_continuous(indexes);
        let tensor = Self::into_continuous(tensor);
        let value = Self::into_continuous(value);
        let tensor = match tensor.can_mut() {
            true => tensor,
            false => tensor.copy(),
        };
        let mut info = build_info(&[&tensor]);
        let mut strides = [0; D];
        let mut current = 1;
        let mut num_elems_per_workgroup = 1;

        tensor
            .shape
            .dims
            .iter()
            .enumerate()
            .rev()
            .filter(|(index, _val)| *index != dim)
            .for_each(|(index, val)| {
                strides[index] = current;
                current *= val;
                num_elems_per_workgroup *= tensor.shape.dims[index];
            });

        strides
            .into_iter()
            .for_each(|stride| info.push(stride as u32));

        info.push(dim as u32);

        let info_buffer = tensor
            .context
            .create_buffer_with_data(bytemuck::cast_slice(&info));

        let kernel = tensor
            .context
            .compile_static::<KernelSettings<Scatter, E, i32, WORKGROUP, 1, 1>>();

        tensor.context.execute(
            WorkGroup::new(
                f32::ceil(num_elems_per_workgroup as f32 / WORKGROUP as f32) as u32,
                1,
                1,
            ),
            kernel,
            &[&tensor.buffer, &indexes.buffer, &value.buffer, &info_buffer],
        );

        tensor
    }
}
