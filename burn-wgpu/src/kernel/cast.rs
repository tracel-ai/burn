use super::{KernelSettings, SourceTemplate, StaticKernelSource, WORKGROUP_DEFAULT};
use crate::{
    compute::StaticKernel, element::JitElement, kernel::elemwise_workgroup, kernel_wgsl,
    tensor::JitTensor, Runtime,
};
use std::{any::TypeId, marker::PhantomData};

kernel_wgsl!(CastRaw, "../template/cast.wgsl");

struct Cast<InputElem: JitElement, OutputElem: JitElement> {
    _i: PhantomData<InputElem>,
    _o: PhantomData<OutputElem>,
}

impl<InputElem: JitElement, OutputElem: JitElement> StaticKernelSource
    for Cast<InputElem, OutputElem>
{
    fn source() -> SourceTemplate {
        CastRaw::source()
            .register("input_elem", InputElem::type_name())
            .register("output_elem", OutputElem::type_name())
    }
}

/// Cast a tensor to the given element type.
pub fn cast<R: Runtime, InputElem: JitElement, OutputElem: JitElement, const D: usize>(
    tensor: JitTensor<R, InputElem, D>,
) -> JitTensor<R, OutputElem, D> {
    if TypeId::of::<InputElem>() == TypeId::of::<OutputElem>() {
        return JitTensor::new(tensor.client, tensor.device, tensor.shape, tensor.handle);
    }

    let num_elems = tensor.shape.num_elements();
    let kernel = StaticKernel::<
        KernelSettings<
            Cast<InputElem, OutputElem>,
            f32,
            i32,
            WORKGROUP_DEFAULT,
            WORKGROUP_DEFAULT,
            1,
        >,
    >::new(elemwise_workgroup(num_elems, WORKGROUP_DEFAULT));

    let handle = tensor
        .client
        .empty(num_elems * core::mem::size_of::<OutputElem>());
    let output = JitTensor::new(
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
    use crate::tests::{TestBackend, TestRuntime};
    use burn_tensor::{Int, Tensor};

    #[test]
    fn should_cast_int_to_float() {
        const START: usize = 0;
        const END: usize = 100;

        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(START as i64..END as i64, &device);
        let tensor_float = cast::<TestRuntime, i32, f32, 1>(tensor.clone().into_primitive());

        let data_int = tensor.into_data();
        let data_float = Tensor::<TestBackend, 1>::from_primitive(tensor_float).into_data();

        for i in START..END {
            assert_eq!(data_int.value[i], i as i32);
            assert_eq!(data_float.value[i], i as f32);
        }
    }
}
