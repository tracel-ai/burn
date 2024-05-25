use crate::{
    calculate_num_elems_dyn_rank,
    language::{CubeType, ExpandElement},
    CubeArg, Runtime, RuntimeArg,
};
use std::marker::PhantomData;

#[derive(new, Clone, Copy)]
pub struct Tensor<E> {
    _val: PhantomData<E>,
}

impl<C: CubeType> CubeType for Tensor<C> {
    type ExpandType = ExpandElement;
}

impl<C: CubeType> CubeArg for Tensor<C> {
    type ArgType<'a, R: Runtime> = TensorHandle<'a, R>;
}

#[derive(new)]
pub struct TensorHandle<'a, R: Runtime> {
    pub handle: &'a burn_compute::server::Handle<R::Server>,
    pub strides: &'a [usize],
    pub shape: &'a [usize],
}

impl<'a, R: Runtime> RuntimeArg<R> for TensorHandle<'a, R> {
    fn register(&self, settings: &mut crate::BindingSettings<R>) {
        settings.arrays.push(self.handle.clone().binding());

        if settings.info.is_empty() {
            settings.info.push(self.strides.len() as u32);
        }

        for s in self.strides.iter() {
            settings.info.push(*s as u32);
        }

        for s in self.shape.iter() {
            settings.info.push(*s as u32);
        }

        if R::require_array_lengths() {
            let len = calculate_num_elems_dyn_rank(self.shape);
            settings.array_lengths.push(len as u32);
        }
    }
}
