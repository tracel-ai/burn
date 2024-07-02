use std::marker::PhantomData;

use crate::{
    compute::{KernelBuilder, KernelLauncher},
    frontend::CubeType,
    ir::{Item, Vectorization},
    unexpanded, KernelSettings, Runtime,
};

use super::{
    ArgSettings, CubePrimitive, ExpandElementTyped, Init, LaunchArg, LaunchArgExpand, TensorHandle,
    UInt,
};

#[derive(new)]
pub struct Array<E> {
    _val: PhantomData<E>,
}

impl<C: CubeType> CubeType for Array<C> {
    type ExpandType = ExpandElementTyped<Array<C>>;
}

impl<C: CubeType> CubeType for &Array<C> {
    type ExpandType = ExpandElementTyped<Array<C>>;
}

impl<C: CubeType> CubeType for &mut Array<C> {
    type ExpandType = ExpandElementTyped<Array<C>>;
}

impl<C: CubeType> Init for ExpandElementTyped<Array<C>> {}

impl<E: CubeType> Array<E> {
    /// Obtain the array length of input
    pub fn len(&self) -> UInt {
        unexpanded!()
    }
}

impl<C: CubePrimitive> LaunchArg for Array<C> {
    type RuntimeArg<'a, R: Runtime> = ArrayArg<'a, R>;
}

impl<C: CubePrimitive> LaunchArgExpand for &Array<C> {
    fn expand(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> ExpandElementTyped<Array<C>> {
        builder
            .input_array(Item::vectorized(C::as_elem(), vectorization))
            .into()
    }
}

impl<C: CubePrimitive> LaunchArgExpand for &mut Array<C> {
    fn expand(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> ExpandElementTyped<Array<C>> {
        builder
            .output_array(Item::vectorized(C::as_elem(), vectorization))
            .into()
    }
}

pub struct ArrayHandle<'a, R: Runtime> {
    pub handle: &'a burn_compute::server::Handle<R::Server>,
    pub length: [usize; 1],
}

pub enum ArrayArg<'a, R: Runtime> {
    Handle {
        handle: ArrayHandle<'a, R>,
        vectorization_factor: u8,
    },
    Alias {
        input_pos: usize,
    },
}

impl<'a, R: Runtime> ArgSettings<R> for ArrayArg<'a, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        if let ArrayArg::Handle {
            handle,
            vectorization_factor: _,
        } = self
        {
            launcher.register_array(handle)
        }
    }

    fn configure_input(&self, position: usize, settings: KernelSettings) -> KernelSettings {
        match self {
            Self::Handle {
                handle: _,
                vectorization_factor,
            } => settings.vectorize_input(position, *vectorization_factor),
            Self::Alias { input_pos: _ } => {
                panic!("Not yet supported, only output can be aliased for now.");
            }
        }
    }

    fn configure_output(&self, position: usize, mut settings: KernelSettings) -> KernelSettings {
        match self {
            Self::Handle {
                handle: _,
                vectorization_factor,
            } => settings.vectorize_output(position, *vectorization_factor),
            Self::Alias { input_pos } => {
                settings.mappings.push(crate::InplaceMapping {
                    pos_input: *input_pos,
                    pos_output: position,
                });
                settings
            }
        }
    }
}

impl<'a, R: Runtime> ArrayArg<'a, R> {
    pub fn new(handle: &'a burn_compute::server::Handle<R::Server>, length: usize) -> Self {
        ArrayArg::Handle {
            handle: ArrayHandle::new(handle, length),
            vectorization_factor: 1,
        }
    }

    pub fn vectorized(
        vectorization_factor: u8,
        handle: &'a burn_compute::server::Handle<R::Server>,
        length: usize,
    ) -> Self {
        ArrayArg::Handle {
            handle: ArrayHandle::new(handle, length),
            vectorization_factor,
        }
    }
}

impl<'a, R: Runtime> ArrayHandle<'a, R> {
    pub fn new(handle: &'a burn_compute::server::Handle<R::Server>, length: usize) -> Self {
        Self {
            handle,
            length: [length],
        }
    }

    pub fn as_tensor(&self) -> TensorHandle<'_, R> {
        let shape = &self.length;

        TensorHandle {
            handle: self.handle,
            strides: &[1],
            shape,
        }
    }
}
