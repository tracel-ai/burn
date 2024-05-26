use crate::{
    dialect::Item,
    language::{CubeType, ExpandElement},
    ArgSettings, CubeElem, KernelLauncher, LaunchArg, Runtime,
};
use std::marker::PhantomData;

#[derive(new, Clone, Copy)]
pub struct Tensor<E> {
    _val: PhantomData<E>,
}

impl<C: CubeType> CubeType for Tensor<C> {
    type ExpandType = ExpandElement;
}

impl<C: CubeElem> LaunchArg for Tensor<C> {
    type RuntimeArg<'a, R: Runtime> = TensorHandle<'a, R>;

    fn compile_input(builder: &mut crate::KernelBuilder) -> ExpandElement {
        builder.input_array(Item::new(C::as_elem()))
    }

    fn compile_output(builder: &mut crate::KernelBuilder) -> ExpandElement {
        builder.output_array(Item::new(C::as_elem()))
    }
}

#[derive(new)]
pub struct TensorHandle<'a, R: Runtime> {
    pub handle: &'a burn_compute::server::Handle<R::Server>,
    pub strides: &'a [usize],
    pub shape: &'a [usize],
}

impl<'a, R: Runtime> ArgSettings<R> for TensorHandle<'a, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        launcher.add_tensor(self)
    }
}
