use super::{KernelSettings, SourceTemplate, StaticKernelSource};
use crate::{
    compute::StaticKernel, element::WgpuElement, kernel::elemwise_workgroup, kernel_wgsl,
    tensor::WgpuTensor,
};
use std::{any::TypeId, marker::PhantomData};

kernel_wgsl!(CastRaw, "../template/cast.wgsl");

struct Cast<InputElem: WgpuElement, OutputElem: WgpuElement> {
    _i: PhantomData<InputElem>,
    _o: PhantomData<OutputElem>,
}

impl<InputElem: WgpuElement, OutputElem: WgpuElement> StaticKernelSource
    for Cast<InputElem, OutputElem>
{
    fn source() -> SourceTemplate {
        CastRaw::source()
            .register("input_elem", InputElem::type_name())
            .register("output_elem", OutputElem::type_name())
    }
}

/// Cast a tensor to the given element type.
pub fn cast<InputElem: WgpuElement, OutputElem: WgpuElement, const D: usize>(
    tensor: WgpuTensor<InputElem, D>,
) -> WgpuTensor<OutputElem, D> {
    if TypeId::of::<InputElem>() == TypeId::of::<OutputElem>() {
        return WgpuTensor::new(tensor.client, tensor.device, tensor.shape, tensor.handle);
    }

    const WORKGROUP: usize = 32;

    let num_elems = tensor.shape.num_elements();
    let kernel = StaticKernel::<
        KernelSettings<Cast<InputElem, OutputElem>, f32, i32, WORKGROUP, WORKGROUP, 1>,
    >::new(elemwise_workgroup(num_elems, WORKGROUP));

    let handle = tensor
        .client
        .empty(num_elems * core::mem::size_of::<OutputElem>());
    let output = WgpuTensor::new(
        tensor.client.clone(),
        tensor.device,
        tensor.shape.clone(),
        handle,
    );

    tensor
        .client
        .execute(Box::new(kernel), &[&tensor.handle, &output.handle]);

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::TestBackend;
    use burn_tensor::{Int, Tensor};

    #[test]
    fn should_cast_int_to_float() {
        const START: usize = 0;
        const END: usize = 100;

        let tensor = Tensor::<TestBackend, 1, Int>::arange(START..END);
        let tensor_float = cast::<i32, f32, 1>(tensor.clone().into_primitive());

        let data_int = tensor.into_data();
        let data_float = Tensor::<TestBackend, 1>::from_primitive(tensor_float).into_data();

        for i in START..END {
            assert_eq!(data_int.value[i], i as i32);
            assert_eq!(data_float.value[i], i as f32);
        }
    }
}
