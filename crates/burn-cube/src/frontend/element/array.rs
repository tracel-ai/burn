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

/// A contiguous array of elements.
#[derive(new)]
pub struct Array<E> {
    _val: PhantomData<E>,
}

impl<C: CubeType> CubeType for Array<C> {
    type ExpandType = ExpandElementTyped<Array<C>>;
}

impl<C: CubeType> Init for ExpandElementTyped<Array<C>> {
    fn init(self, _context: &mut crate::prelude::CubeContext) -> Self {
        // The type can't be deeply cloned/copied.
        self
    }
}

impl<E: CubeType> Array<E> {
    /// Obtain the array length
    pub fn len(&self) -> UInt {
        unexpanded!()
    }
}

impl<C: CubePrimitive> LaunchArg for Array<C> {
    type RuntimeArg<'a, R: Runtime> = ArrayArg<'a, R>;
}

impl<C: CubePrimitive> LaunchArgExpand for Array<C> {
    fn expand(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> ExpandElementTyped<Array<C>> {
        builder
            .input_array(Item::vectorized(C::as_elem(), vectorization))
            .into()
    }
    fn expand_output(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> ExpandElementTyped<Array<C>> {
        builder
            .output_array(Item::vectorized(C::as_elem(), vectorization))
            .into()
    }
}

/// Tensor representation with a reference to the [server handle](burn_compute::server::Handle).
pub struct ArrayHandle<'a, R: Runtime> {
    pub handle: &'a burn_compute::server::Handle<R::Server>,
    pub length: [usize; 1],
}

pub enum ArrayArg<'a, R: Runtime> {
    /// The array is passed with an array handle.
    Handle {
        /// The array handle.
        handle: ArrayHandle<'a, R>,
        /// The vectorization factor.
        vectorization_factor: u8,
    },
    /// The array is aliasing another input array.
    Alias {
        /// The position of the input array.
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
    /// Create a new array argument.
    ///
    /// Equivalent to using the [vectorized constructor](Self::vectorized) with a vectorization
    /// factor of 1.
    pub fn new(handle: &'a burn_compute::server::Handle<R::Server>, length: usize) -> Self {
        ArrayArg::Handle {
            handle: ArrayHandle::new(handle, length),
            vectorization_factor: 1,
        }
    }
    /// Create a new array argument specified with its vectorization factor.
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
