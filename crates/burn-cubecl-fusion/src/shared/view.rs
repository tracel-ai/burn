use cubecl::{
    CubeType,
    io::read_masked,
    ir::StorageType,
    prelude::{barrier::BarrierExpand, *},
    std::{
        CubeOption, CubeOptionExpand,
        tensor::{
            ViewOperations, ViewOperationsExpand, ViewOperationsMut, ViewOperationsMutExpand,
            layout::Coords1d,
        },
    },
};

use crate::shared::{
    DYN_ELEM_ID,
    io::{
        Transform, global_buffer_len, global_line_size, input_as_slice, read_input,
        read_input_window, ref_buffer_len, ref_len,
    },
    ir::{Arg, FuseBlockConfig, GlobalArgs, LayoutInfo, LocalArgs},
    kernel::fuse_on_write,
};

#[allow(dead_code, reason = "only used in expand")]
#[derive(CubeType)]
pub struct GlobalInput {
    inputs: GlobalArgs,
    locals: LocalArgs,
    #[cube(comptime)]
    pos: u32,
    #[cube(comptime)]
    ty: StorageType,
    #[cube(comptime)]
    layout: LayoutInfo,
    #[cube(comptime)]
    config: FuseBlockConfig,
    #[cube(comptime)]
    transform: Option<Transform>,
}

#[cube]
impl GlobalInput {
    pub fn new(
        inputs: &GlobalArgs,
        locals: &LocalArgs,
        #[comptime] arg: Arg,
        #[comptime] config: FuseBlockConfig,
        #[comptime] transform: Option<Transform>,
    ) -> GlobalInput {
        let (pos, ty, layout) = comptime![match arg {
            Arg::Input(pos, prec, layout) => (pos, prec.into_type(), layout),
            _ => unreachable!("Must be concrete input"),
        }];

        GlobalInput {
            inputs: inputs.clone(),
            locals: locals.clone(),
            pos,
            ty,
            layout,
            config,
            transform,
        }
    }
}

impl<E: CubePrimitive> ViewOperations<Line<E>, Coords1d> for GlobalInput {}
impl<E: CubePrimitive> ViewOperationsExpand<Line<E>, Coords1d> for GlobalInputExpand {
    #[allow(clippy::too_many_arguments)]
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
    ) -> <Line<E> as CubeType>::ExpandType {
        ViewOperationsExpand::<Line<E>, Coords1d>::__expand_read_unchecked_method(self, scope, pos)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_checked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
    ) -> <Line<E> as CubeType>::ExpandType {
        let zero = Line::<E>::__expand_cast_from(scope, 0.into());
        self.__expand_read_masked_method(scope, pos, zero)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_masked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
        value: <Line<E> as CubeType>::ExpandType,
    ) -> <Line<E> as CubeType>::ExpandType {
        let in_bounds = ViewOperationsExpand::<Line<E>, Coords1d>::__expand_is_in_bounds_method(
            self,
            scope,
            pos.clone(),
        );
        scope.register_type::<NumericExpand<DYN_ELEM_ID>>(self.ty);
        let slice = input_as_slice::expand(scope, self.inputs.clone(), self.pos);
        read_masked::expand::<Line<E>>(scope, in_bounds, slice, pos, value)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
    ) -> <Line<E> as CubeType>::ExpandType {
        read_input::expand(
            scope,
            self.inputs.clone(),
            self.locals.clone(),
            self.pos,
            pos,
            self.layout,
            self.config.clone(),
            self.transform.clone(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_to_linear_slice_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<E>, ReadOnly> {
        scope.register_type::<NumericExpand<DYN_ELEM_ID>>(self.ty);
        let end = add::expand(scope, end.clone(), 1.into());
        read_input_window::expand(scope, self.inputs.clone(), self.pos, pos, end)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> CubeOptionExpand<TensorMap<Line<E>>> {
        CubeOption::__expand_new_None(scope)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_tensor_map_load_method(
        &self,
        _scope: &mut Scope,
        _barrier: BarrierExpand,
        _shared_memory: SliceExpand<Line<E>, ReadWrite>,
        _pos: ExpandElementTyped<u32>,
    ) {
        panic!("Not a tensor map")
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_shape_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        global_buffer_len::expand(scope, self.inputs.clone(), self.pos)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<bool> {
        let buffer_len = global_buffer_len::expand(scope, self.inputs.clone(), self.pos);
        lt::expand(scope, pos, buffer_len)
    }
}

impl Lined for GlobalInput {}
impl LinedExpand for GlobalInputExpand {
    fn line_size(&self) -> u32 {
        let mut temp_scope = Scope::root(false);
        global_line_size::expand(&mut temp_scope, self.inputs.clone(), self.pos)
    }
}

#[allow(dead_code, reason = "only used in expand")]
#[derive(CubeType)]
pub struct FusedOutput {
    inputs: GlobalArgs,
    outputs: GlobalArgs,
    locals: LocalArgs,
    arg: Arg,
    #[cube(comptime)]
    config: FuseBlockConfig,
}

#[cube]
impl FusedOutput {
    pub fn new(
        inputs: &GlobalArgs,
        outputs: &mut GlobalArgs,
        locals: &mut LocalArgs,
        arg: Arg,
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

impl<E: CubePrimitive> ViewOperations<Line<E>, Coords1d> for FusedOutput {}
impl<E: CubePrimitive> ViewOperationsExpand<Line<E>, Coords1d> for FusedOutputExpand {
    #[allow(clippy::too_many_arguments)]
    fn __expand_read_method(
        &self,
        _scope: &mut Scope,
        _pos: ExpandElementTyped<u32>,
    ) -> <Line<E> as CubeType>::ExpandType {
        todo!()
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_checked_method(
        &self,
        _scope: &mut Scope,
        _pos: ExpandElementTyped<u32>,
    ) -> <Line<E> as CubeType>::ExpandType {
        todo!()
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_masked_method(
        &self,
        _scope: &mut Scope,
        _pos: ExpandElementTyped<u32>,
        _value: <Line<E> as CubeType>::ExpandType,
    ) -> <Line<E> as CubeType>::ExpandType {
        todo!()
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_unchecked_method(
        &self,
        _scope: &mut Scope,
        _pos: ExpandElementTyped<u32>,
    ) -> <Line<E> as CubeType>::ExpandType {
        todo!()
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_to_linear_slice_method(
        &self,
        _scope: &mut Scope,
        _pos: ExpandElementTyped<u32>,
        _size: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<E>, ReadOnly> {
        todo!()
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> CubeOptionExpand<TensorMap<Line<E>>> {
        CubeOption::__expand_new_None(scope)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_tensor_map_load_method(
        &self,
        _scope: &mut Scope,
        _barrier: BarrierExpand,
        _shared_memory: SliceExpand<Line<E>, ReadWrite>,
        _pos: ExpandElementTyped<u32>,
    ) {
        panic!("Not a tensor map")
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_shape_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        ref_len::expand(
            scope,
            self.inputs.clone(),
            self.outputs.clone(),
            self.locals.clone(),
            self.config.clone(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<bool> {
        let buffer_len = ref_buffer_len::expand(
            scope,
            self.inputs.clone(),
            self.outputs.clone(),
            self.locals.clone(),
            self.config.clone(),
        );
        lt::expand(scope, pos, buffer_len)
    }
}

impl<E: CubePrimitive> ViewOperationsMut<Line<E>, Coords1d> for FusedOutput {}
impl<E: CubePrimitive> ViewOperationsMutExpand<Line<E>, Coords1d> for FusedOutputExpand {
    #[allow(clippy::too_many_arguments)]
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
        value: <Line<E> as CubeType>::ExpandType,
    ) {
        let values = Registry::<Arg, Line<E>>::__expand_new(scope);
        let mut args = comptime![Sequence::<Arg>::new()];

        values
            .clone()
            .__expand_insert_method(scope, comptime![self.arg.clone()], value);
        comptime![args.push(self.arg.clone())];

        fuse_on_write::expand(
            scope,
            self.inputs.clone(),
            self.outputs.clone(),
            self.locals.clone(),
            pos,
            values,
            args,
            self.config.clone(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_write_checked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
        value: <Line<E> as CubeType>::ExpandType,
    ) {
        let in_bounds = ViewOperationsExpand::<Line<E>, Coords1d>::__expand_is_in_bounds_method(
            self,
            scope,
            pos.clone(),
        );
        if_expand(scope, in_bounds.into(), |scope| {
            self.__expand_write_method(scope, pos, value);
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_to_linear_slice_mut_method(
        &self,
        _scope: &mut Scope,
        _pos: ExpandElementTyped<u32>,
        _size: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<E>, ReadWrite> {
        todo!("Not yet supported")
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_tensor_map_store_method(
        &self,
        _scope: &mut Scope,
        _shared_memory: SliceExpand<Line<E>, ReadOnly>,
        _pos: ExpandElementTyped<u32>,
    ) {
        panic!("Not a tensor map")
    }
}

impl Lined for FusedOutput {}
impl LinedExpand for FusedOutputExpand {
    fn line_size(&self) -> u32 {
        self.locals.ref_line_size
    }
}
