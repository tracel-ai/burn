use crate::engine::codegen::{DynElem, DynSize, io::set_polyfill_typed};

use super::{
    io::{
        global_buffer_len, global_vector_size, input_as_slice, read_input_window, ref_buffer_len,
        ref_len,
    },
    ir::{FuseArg, FuseBlockConfig, GlobalArgs, LocalArgs},
    kernel::fuse_on_write,
};
use cubecl::{
    CubeType,
    io::read_masked,
    prelude::{barrier::BarrierExpand, *},
    std::tensor::{
        ViewOperations, ViewOperationsExpand, ViewOperationsMut, ViewOperationsMutExpand,
        layout::Coords1d,
    },
};

#[allow(dead_code, reason = "only used in expand")]
#[derive(CubeType)]
pub struct GlobalInput {
    inputs: GlobalArgs,
    #[cube(comptime)]
    pos: usize,
}

#[cube]
impl GlobalInput {
    pub fn new(inputs: &GlobalArgs, #[comptime] arg: FuseArg) -> GlobalInput {
        let pos = comptime![match arg {
            FuseArg::Input(pos, ..) => pos,
            _ => unreachable!("Must be concrete input"),
        }];

        GlobalInput {
            inputs: inputs.clone(),
            pos,
        }
    }
}

impl<E: CubePrimitive> ViewOperations<E, Coords1d> for GlobalInput {}
impl<E: CubePrimitive> ViewOperationsExpand<E, Coords1d> for GlobalInputExpand {
    #[allow(clippy::too_many_arguments)]
    fn __expand_read_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> <E as CubeType>::ExpandType {
        ViewOperationsExpand::<E, Coords1d>::__expand_read_unchecked_method(self, scope, pos)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_checked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> <E as CubeType>::ExpandType {
        let zero = E::__expand_cast_from(scope, 0.into());
        ViewOperationsExpand::<E, Coords1d>::__expand_read_masked_method(self, scope, pos, zero)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_masked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        value: <E as CubeType>::ExpandType,
    ) -> <E as CubeType>::ExpandType {
        let in_bounds =
            ViewOperationsExpand::<E, Coords1d>::__expand_is_in_bounds_method(self, scope, pos);
        set_polyfill_typed::expand::<E, DynElem, DynSize>(scope);
        let slice = input_as_slice::expand(scope, &self.inputs, self.pos);
        read_masked::expand::<E>(scope, in_bounds, slice, pos, value)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> <E as CubeType>::ExpandType {
        set_polyfill_typed::expand::<E, DynElem, DynSize>(scope);
        // The view's layout already maps coordinates to physical buffer offsets.
        // Applying the fusion reference layout here would remap the offset twice.
        let slice = input_as_slice::expand::<E>(scope, &self.inputs, self.pos);
        slice.__expand_read_unchecked_method(scope, pos)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_as_linear_slice_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &SliceExpand<E> {
        set_polyfill_typed::expand::<E, DynElem, DynSize>(scope);
        let end = end.__expand_add_method(scope, 1usize.into_expand(scope));
        read_input_window::expand(scope, &self.inputs, self.pos, pos, end)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_tensor_map_load_method(
        &self,
        _scope: &Scope,
        _barrier: &BarrierExpand,
        _shared_memory: &mut SliceExpand<E>,
        _pos: NativeExpand<usize>,
    ) {
        panic!("Not a tensor map")
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_shape_method(&self, scope: &Scope) -> NativeExpand<usize> {
        global_buffer_len::expand(scope, &self.inputs, self.pos)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> NativeExpand<bool> {
        let buffer_len = global_buffer_len::expand(scope, &self.inputs, self.pos);
        pos.__expand_lt_method(scope, &buffer_len)
    }
}

impl Vectorized for GlobalInput {}
impl VectorizedExpand for GlobalInputExpand {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        global_vector_size::expand(scope, &self.inputs, self.pos)
    }
}

#[allow(dead_code, reason = "only used in expand")]
#[derive(CubeType)]
pub struct FusedOutput {
    inputs: GlobalArgs,
    outputs: GlobalArgs,
    locals: LocalArgs,
    arg: FuseArg,
    #[cube(comptime)]
    config: FuseBlockConfig,
}

#[cube]
impl FusedOutput {
    pub fn new(
        inputs: &GlobalArgs,
        outputs: &mut GlobalArgs,
        locals: &mut LocalArgs,
        arg: FuseArg,
        #[comptime] config: FuseBlockConfig,
    ) -> Self {
        FusedOutput {
            inputs: inputs.clone(),
            outputs: outputs.clone(),
            locals: locals.clone(),
            arg,
            config,
        }
    }
}

impl<E: CubePrimitive> ViewOperations<E, Coords1d> for FusedOutput {}
impl<E: CubePrimitive> ViewOperationsExpand<E, Coords1d> for FusedOutputExpand {
    #[allow(clippy::too_many_arguments)]
    fn __expand_read_method(
        &self,
        _scope: &Scope,
        _pos: NativeExpand<usize>,
    ) -> <E as CubeType>::ExpandType {
        todo!()
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_checked_method(
        &self,
        _scope: &Scope,
        _pos: NativeExpand<usize>,
    ) -> <E as CubeType>::ExpandType {
        todo!()
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_masked_method(
        &self,
        _scope: &Scope,
        _pos: NativeExpand<usize>,
        _value: <E as CubeType>::ExpandType,
    ) -> <E as CubeType>::ExpandType {
        todo!()
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_unchecked_method(
        &self,
        _scope: &Scope,
        _pos: NativeExpand<usize>,
    ) -> <E as CubeType>::ExpandType {
        todo!()
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_as_linear_slice_method(
        &self,
        _scope: &Scope,
        _pos: NativeExpand<usize>,
        _size: NativeExpand<usize>,
    ) -> &SliceExpand<E> {
        todo!()
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_tensor_map_load_method(
        &self,
        _scope: &Scope,
        _barrier: &BarrierExpand,
        _shared_memory: &mut SliceExpand<E>,
        _pos: NativeExpand<usize>,
    ) {
        panic!("Not a tensor map")
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_shape_method(&self, scope: &Scope) -> NativeExpand<usize> {
        ref_len::expand(
            scope,
            &self.inputs,
            &self.outputs,
            &self.locals,
            &self.config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> NativeExpand<bool> {
        let buffer_len = ref_buffer_len::expand(
            scope,
            &self.inputs,
            &self.outputs,
            &self.locals,
            &self.config,
        );
        pos.__expand_lt_method(scope, &buffer_len)
    }
}

impl<E: CubePrimitive> ViewOperationsMut<E, Coords1d> for FusedOutput {}
impl<E: CubePrimitive> ViewOperationsMutExpand<E, Coords1d> for FusedOutputExpand {
    #[allow(clippy::too_many_arguments)]
    fn __expand_write_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        value: <E as CubeType>::ExpandType,
    ) {
        let values = Registry::<FuseArg, Vector<E::Scalar, E::Size>>::__expand_new(scope);
        let mut args = comptime![Vec::<FuseArg>::new()];

        let value = Vector::__expand_cast_from(scope, value);
        values
            .clone()
            .__expand_insert_method(scope, comptime![self.arg.clone()], value);
        comptime![args.push(self.arg.clone())];

        let mut outputs = self.outputs.clone();
        let mut locals = self.locals.clone();

        fuse_on_write::expand(
            scope,
            &self.inputs,
            &mut outputs,
            &mut locals,
            pos,
            values,
            args,
            &self.config,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_write_checked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        value: <E as CubeType>::ExpandType,
    ) {
        let in_bounds =
            ViewOperationsExpand::<E, Coords1d>::__expand_is_in_bounds_method(self, scope, pos);
        if_expand(scope, in_bounds, |scope| {
            ViewOperationsMutExpand::<E, Coords1d>::__expand_write_method(self, scope, pos, value);
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_as_linear_slice_mut_method(
        &self,
        _scope: &Scope,
        _pos: NativeExpand<usize>,
        _size: NativeExpand<usize>,
    ) -> &mut SliceExpand<E> {
        todo!("Not yet supported")
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_tensor_map_store_method(
        &self,
        _scope: &Scope,
        _shared_memory: &SliceExpand<E>,
        _pos: NativeExpand<usize>,
    ) {
        panic!("Not a tensor map")
    }
}

impl Vectorized for FusedOutput {}
impl VectorizedExpand for FusedOutputExpand {
    fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        self.locals.ref_vector_size
    }
}
