use super::{ExpandElementTyped, Init, LaunchArgExpand};
use crate::{
    frontend::{
        indexation::Index, ArgSettings, CubeContext, CubePrimitive, CubeType, ExpandElement, UInt,
    },
    ir::{Elem, Item, Metadata, Variable, Vectorization},
    prelude::{KernelBuilder, KernelLauncher},
    unexpanded, KernelSettings, LaunchArg, Runtime,
};
use std::marker::PhantomData;

/// The tensor type is similar to the [array type](crate::prelude::Array), however it comes with more
/// metadata such as [stride](Tensor::stride) and [shape](Tensor::shape).
#[derive(new)]
pub struct Tensor<T: CubeType> {
    _val: PhantomData<T>,
}

impl<T: CubeType> CubeType for Tensor<T> {
    type ExpandType = ExpandElementTyped<Tensor<T>>;
}

impl<T: CubeType> Init for ExpandElementTyped<Tensor<T>> {
    fn init(self, _context: &mut crate::prelude::CubeContext) -> Self {
        // The type can't be deeply cloned/copied.
        self
    }
}

impl<C: CubePrimitive> LaunchArgExpand for Tensor<C> {
    fn expand(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> ExpandElementTyped<Tensor<C>> {
        builder
            .input_array(Item::vectorized(C::as_elem(), vectorization))
            .into()
    }
    fn expand_output(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> ExpandElementTyped<Tensor<C>> {
        builder
            .output_array(Item::vectorized(C::as_elem(), vectorization))
            .into()
    }
}

impl<C: CubePrimitive> LaunchArg for Tensor<C> {
    type RuntimeArg<'a, R: Runtime> = TensorArg<'a, R>;
}

/// Tensor representation with a reference to the [server handle](burn_compute::server::Handle),
/// the strides and the shape.
#[derive(new)]
pub struct TensorHandle<'a, R: Runtime> {
    pub handle: &'a burn_compute::server::Handle<R::Server>,
    pub strides: &'a [usize],
    pub shape: &'a [usize],
}

/// Argument to be used for [tensors](Tensor) passed as arguments to kernels.
pub enum TensorArg<'a, R: Runtime> {
    /// The tensor is passed with a tensor handle.
    Handle {
        /// The tensor handle.
        handle: TensorHandle<'a, R>,
        /// The vectorization factor.
        vectorization_factor: u8,
    },
    /// The tensor is aliasing another input tensor.
    Alias {
        /// The position of the input tensor.
        input_pos: usize,
    },
}

impl<'a, R: Runtime> TensorArg<'a, R> {
    /// Create a new tensor argument.
    ///
    /// Equivalent to using the [vectorized constructor](Self::vectorized) with a vectorization
    /// factor of 1.
    pub fn new(
        handle: &'a burn_compute::server::Handle<R::Server>,
        strides: &'a [usize],
        shape: &'a [usize],
    ) -> Self {
        Self::Handle {
            handle: TensorHandle::new(handle, strides, shape),
            vectorization_factor: 1,
        }
    }
    /// Create a new tensor argument specified with its vectorization factor.
    pub fn vectorized(
        factor: u8,
        handle: &'a burn_compute::server::Handle<R::Server>,
        strides: &'a [usize],
        shape: &'a [usize],
    ) -> Self {
        Self::Handle {
            handle: TensorHandle::new(handle, strides, shape),
            vectorization_factor: factor,
        }
    }
    pub fn alias(position: usize) -> Self {
        Self::Alias {
            input_pos: position,
        }
    }
}

impl<'a, R: Runtime> ArgSettings<R> for TensorArg<'a, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        if let TensorArg::Handle {
            handle,
            vectorization_factor: _,
        } = self
        {
            launcher.register_tensor(handle)
        }
    }

    fn configure_input(&self, position: usize, settings: KernelSettings) -> KernelSettings {
        match self {
            TensorArg::Handle {
                handle: _,
                vectorization_factor,
            } => settings.vectorize_input(position, *vectorization_factor),
            TensorArg::Alias { input_pos: _ } => {
                panic!("Not yet supported, only output can be aliased for now.");
            }
        }
    }

    fn configure_output(&self, position: usize, mut settings: KernelSettings) -> KernelSettings {
        match self {
            TensorArg::Handle {
                handle: _,
                vectorization_factor,
            } => settings.vectorize_output(position, *vectorization_factor),
            TensorArg::Alias { input_pos } => {
                settings.mappings.push(crate::InplaceMapping {
                    pos_input: *input_pos,
                    pos_output: position,
                });
                settings
            }
        }
    }
}

impl<T: CubeType> Tensor<T> {
    /// Obtain the stride of input at dimension dim
    pub fn stride<C: Index>(&self, _dim: C) -> UInt {
        unexpanded!()
    }

    /// Obtain the shape of input at dimension dim
    pub fn shape<C: Index>(&self, _dim: C) -> UInt {
        unexpanded!()
    }

    /// The length of the buffer representing the tensor.
    ///
    /// # Warning
    ///
    /// The length will be affected by the vectorization factor. To obtain the number of elements,
    /// you should multiply the length by the vectorization factor.
    pub fn len(&self) -> UInt {
        unexpanded!()
    }

    /// Returns the rank of the tensor.
    pub fn rank(&self) -> UInt {
        unexpanded!()
    }
}

impl<T> ExpandElementTyped<T> {
    // Expanded version of stride
    pub fn stride_expand<C: Index>(self, context: &mut CubeContext, dim: C) -> ExpandElement {
        let out = context.create_local(Item::new(Elem::UInt));
        context.register(Metadata::Stride {
            dim: dim.value(),
            var: self.expand.into(),
            out: out.clone().into(),
        });
        out
    }

    // Expanded version of shape
    pub fn shape_expand<C: Index>(self, context: &mut CubeContext, dim: C) -> ExpandElement {
        let out = context.create_local(Item::new(Elem::UInt));
        context.register(Metadata::Shape {
            dim: dim.value(),
            var: self.expand.into(),
            out: out.clone().into(),
        });
        out
    }

    // Expanded version of len
    pub fn len_expand(self, context: &mut CubeContext) -> ExpandElement {
        let out = context.create_local(Item::new(Elem::UInt));
        context.register(Metadata::Length {
            var: self.expand.into(),
            out: out.clone().into(),
        });
        out
    }

    // Expanded version of rank.
    pub fn rank_expand(self, _context: &mut CubeContext) -> ExpandElement {
        ExpandElement::Plain(Variable::Rank)
    }
}
