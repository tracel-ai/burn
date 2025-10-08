use cubecl::{
    CubeType,
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
    io::{ref_buffer_len, ref_len},
    ir::{Arg, FuseBlockConfig, GlobalArgs, LocalArgs},
    kernel::fuse_on_write,
};

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
        panic!("Can't read from output")
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_checked_method(
        &self,
        _scope: &mut Scope,
        _pos: ExpandElementTyped<u32>,
    ) -> <Line<E> as CubeType>::ExpandType {
        panic!("Can't read from output")
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_masked_method(
        &self,
        _scope: &mut Scope,
        _pos: ExpandElementTyped<u32>,
        _value: <Line<E> as CubeType>::ExpandType,
    ) -> <Line<E> as CubeType>::ExpandType {
        panic!("Can't read from output")
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_read_unchecked_method(
        &self,
        _scope: &mut Scope,
        _pos: ExpandElementTyped<u32>,
    ) -> <Line<E> as CubeType>::ExpandType {
        panic!("Can't read from output")
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_to_linear_slice_method(
        &self,
        _scope: &mut Scope,
        _pos: ExpandElementTyped<u32>,
        _size: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<E>, ReadOnly> {
        panic!("Can't read from output")
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
