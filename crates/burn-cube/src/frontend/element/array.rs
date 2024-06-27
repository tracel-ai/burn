use std::marker::PhantomData;

use crate::{
    compute::{KernelBuilder, KernelLauncher},
    frontend::{CubeType, ExpandElement},
    ir::{Item, Vectorization},
    unexpanded, Runtime,
};
use crate::{
    frontend::{indexation::Index, CubeContext},
    prelude::{assign, index, index_assign, Comptime},
};

use super::{ArgSettings, CubePrimitive, LaunchArg, LaunchArgExpand, TensorHandle, UInt};

pub struct Array<E> {
    _val: PhantomData<E>,
}

impl<T: CubePrimitive> CubeType for Array<T> {
    type ExpandType = ExpandElement;
}

impl<T: CubePrimitive + Clone> Array<T> {
    pub fn new<S: Index>(_size: S) -> Self {
        Array { _val: PhantomData }
    }

    pub fn new_expand<S: Index>(
        context: &mut CubeContext,
        size: S,
    ) -> <Self as CubeType>::ExpandType {
        let size = size.value();
        let size = match size {
            crate::ir::Variable::ConstantScalar(val, _) => val as u32,
            _ => panic!("Array need constant initialization value"),
        };
        context.create_local_array(Item::new(T::as_elem()), size)
    }

    pub fn vectorized<S: Index>(_size: S, _vectorization_factor: UInt) -> Self {
        Array { _val: PhantomData }
    }

    pub fn vectorized_expand<S: Index>(
        context: &mut CubeContext,
        size: S,
        vectorization_factor: UInt,
    ) -> <Self as CubeType>::ExpandType {
        let size = size.value();
        let size = match size {
            crate::ir::Variable::ConstantScalar(val, _) => val as u32,
            _ => panic!("Shared memory need constant initialization value"),
        };
        context.create_local_array(
            Item::vectorized(T::as_elem(), vectorization_factor.val as u8),
            size,
        )
    }

    pub fn to_vectorized(self, _vectorization_factor: Comptime<UInt>) -> T {
        unexpanded!()
    }
}

impl ExpandElement {
    pub fn to_vectorized_expand(
        self,
        context: &mut CubeContext,
        vectorization_factor: UInt,
    ) -> ExpandElement {
        let factor = vectorization_factor.val;
        let var = *self;
        let mut new_var = context.create_local(Item::vectorized(var.item().elem(), factor as u8));
        if vectorization_factor.val == 1 {
            let element = index::expand(context, self.clone(), 0u32);
            assign::expand(context, element, new_var.clone());
        } else {
            for i in 0..factor {
                let element = index::expand(context, self.clone(), i);
                new_var = index_assign::expand(context, new_var, i, element);
            }
        }
        new_var
    }
}

impl<C: CubeType> CubeType for &Array<C> {
    type ExpandType = ExpandElement;
}

impl<C: CubeType> CubeType for &mut Array<C> {
    type ExpandType = ExpandElement;
}

impl<E: CubeType> Array<E> {
    /// Obtain the array length of input
    pub fn len(&self) -> UInt {
        unexpanded!()
    }
}

impl<C: CubePrimitive> LaunchArg for Array<C> {
    type RuntimeArg<'a, R: Runtime> = ArrayHandle<'a, R>;
}

impl<C: CubePrimitive> LaunchArgExpand for &Array<C> {
    fn expand(builder: &mut KernelBuilder, vectorization: Vectorization) -> ExpandElement {
        builder.input_array(Item::vectorized(C::as_elem(), vectorization))
    }
}

impl<C: CubePrimitive> LaunchArgExpand for &mut Array<C> {
    fn expand(builder: &mut KernelBuilder, vectorization: Vectorization) -> ExpandElement {
        builder.output_array(Item::vectorized(C::as_elem(), vectorization))
    }
}

pub struct ArrayHandle<'a, R: Runtime> {
    pub handle: &'a burn_compute::server::Handle<R::Server>,
    pub length: [usize; 1],
}

impl<'a, R: Runtime> ArgSettings<R> for ArrayHandle<'a, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        launcher.register_array(self)
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
