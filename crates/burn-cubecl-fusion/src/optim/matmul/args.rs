use crate::engine::codegen::{
    io::ref_line_size,
    ir::{FuseArg, FuseBlockConfig, FuseType, GlobalArgs, LocalArgs},
    kernel::init_locals,
    view::{FusedOutput, GlobalInput, GlobalInputExpand},
};
use cubecl::{
    intrinsic,
    prelude::*,
    quant::scheme::{QuantLevel, QuantScheme},
    std::{
        CubeOption, CubeOptionExpand, FastDivmod,
        quant::{
            RunWithQuantType,
            view::{QuantizedView, run_with_quant_type},
        },
        tensor::{
            View, ViewExpand,
            layout::{Coords1d, Coords2d, Coords3d, VirtualLayout},
        },
    },
};
use cubek::matmul::{
    components::global::memory::{
        BatchLayout, BlockScaledLayout, GlobalLayout, GlobalLayoutConfig, GlobalLayoutExpand,
        GlobalScaleLayout, GlobalScaleLayoutExpand, NoopLayout,
    },
    definition::MatrixLayout,
    launch::MatmulArgs,
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct FusedMatmulArgs;

#[derive(CubeLaunch, CubeType)]
pub struct FusedMatmulInput {
    global: GlobalArgs,
    #[cube(comptime)]
    config: FuseBlockConfig,
    #[cube(comptime)]
    a: MatmulArg,
    #[cube(comptime)]
    b: MatmulArg,
    #[cube(comptime)]
    c: Option<MatmulArg>,
    #[cube(comptime)]
    out: FuseArg,
}

#[cube]
impl MatmulArgs for FusedMatmulArgs {
    type Output<EO: Numeric> = GlobalArgs;
    type Input<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = FusedMatmulInput;
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = FusedMatmulState;

    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        inputs: &Self::Input<Lhs, Rhs, EO>,
        outputs: &mut Self::Output<EO>,
        #[comptime] lhs_layout_config: GlobalLayoutConfig,
        #[comptime] rhs_layout_config: GlobalLayoutConfig,
        #[comptime] out_layout_config: GlobalLayoutConfig,
    ) -> Self::State<Lhs, Rhs, EO> {
        let mut locals = init_locals(&inputs.global, outputs, &inputs.config);
        let rank = comptime![inputs.config.rank];

        let mut batch_shape = Sequence::new();
        let mut batch_strides_out = Sequence::new();

        #[unroll]
        for i in 0..rank - 2 {
            batch_shape.push(FastDivmod::new_Fallback(locals.ref_shape[i]));
            batch_strides_out.push(locals.ref_strides[i]);
        }

        let batch_lhs = input_batch_layout(
            &inputs.global,
            &batch_shape,
            comptime![inputs.a.clone()],
            comptime![inputs.config.clone()],
        );
        let batch_rhs = input_batch_layout(
            &inputs.global,
            &batch_shape,
            comptime![inputs.b.clone()],
            comptime![inputs.config.clone()],
        );
        let batch_acc = match comptime![inputs.c.clone()] {
            Option::Some(c) => CubeOption::new_Some(input_batch_layout(
                &inputs.global,
                &batch_shape,
                comptime![c],
                comptime![inputs.config.clone()],
            )),
            Option::None => CubeOption::new_None(),
        };
        let batch_out = BatchLayout::new(batch_strides_out, batch_shape.clone());

        FusedMatmulState::new(
            inputs,
            outputs,
            &mut locals,
            batch_lhs,
            batch_rhs,
            batch_acc,
            VirtualLayout::new::<BatchLayout>(batch_out),
            batch_shape,
            &inputs.config,
            lhs_layout_config,
            rhs_layout_config,
            out_layout_config,
        )
    }

    fn view_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Lhs>, Coords3d> {
        global_view(
            &state.inputs,
            &state.locals,
            &state.batch_shape,
            comptime![state.a.clone()],
            comptime![state.config.clone()],
            comptime![state.lhs_layout_config],
        )
    }

    fn batch_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: u32,
    ) -> u32 {
        state.a_batch.to_source_pos(batch)
    }

    fn view_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Rhs>, Coords3d> {
        global_view(
            &state.inputs,
            &state.locals,
            &state.batch_shape,
            comptime![state.b.clone()],
            comptime![state.config.clone()],
            comptime![state.rhs_layout_config],
        )
    }

    fn batch_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: u32,
    ) -> u32 {
        state.b_batch.to_source_pos(batch)
    }

    fn view_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<View<Line<EO>, Coords3d>> {
        match comptime![state.c.clone()] {
            Option::Some(c) => {
                let view = global_view(
                    &state.inputs,
                    &state.locals,
                    &state.batch_shape,
                    c,
                    comptime![state.config.clone()],
                    comptime![state.out_layout_config],
                );
                CubeOption::new_Some(view)
            }
            Option::None => CubeOption::new_None(),
        }
    }

    fn batch_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: u32,
    ) -> u32 {
        match state.c_batch {
            CubeOption::Some(c_batch) => c_batch.to_source_pos(batch),
            CubeOption::None => batch,
        }
    }

    fn view_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &mut Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<EO>, Coords3d, ReadWrite> {
        let rank = comptime![state.config.rank];

        let shape_row = state.locals.ref_shape[rank - 2];
        let shape_col = state.locals.ref_shape[rank - 1];

        let stride_row = state.locals.ref_strides[rank - 2];
        let stride_col = state.locals.ref_strides[rank - 1];

        let layout = GlobalLayout::new(
            VirtualLayout::new::<NoopLayout>(NoopLayout::new()),
            shape_row,
            shape_col,
            stride_row,
            stride_col,
            ref_line_size(&state.locals),
            1u32,
            state.out_layout_config,
        );
        let mut buffer = FusedOutput::new(
            &state.inputs,
            &mut state.outputs,
            &mut state.locals,
            comptime![state.out.clone()],
            comptime![state.config.clone()],
        );
        View::new_mut::<FusedOutput, Coords1d>(&mut buffer, layout)
    }

    fn batch_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: u32,
    ) -> u32 {
        state.out_batch.to_source_pos(batch)
    }
}

#[cube]
fn global_view<E: Numeric>(
    inputs: &GlobalArgs,
    locals: &LocalArgs,
    batch_shape: &Sequence<FastDivmod>,
    #[comptime] arg: MatmulArg,
    #[comptime] config: FuseBlockConfig,
    #[comptime] layout_config: GlobalLayoutConfig,
) -> View<Line<E>, Coords3d> {
    let rank = comptime![config.rank];
    let data = comptime![arg.data().clone()];
    let data_tensor = match comptime![data.clone()] {
        FuseArg::Input(pos, ..) => inputs.tensors.index(pos),
        _ => panic!("Input must be concrete"),
    };

    let mut shape_row = data_tensor.tensor.shape(rank - 2);
    let mut shape_col = data_tensor.tensor.shape(rank - 1);
    let mut packing = comptime![1u32];

    if comptime![arg.scheme().is_some()] {
        let scheme = comptime![arg.scheme().unwrap()];
        let num_quants = comptime![scheme.num_quants() as u32];
        comptime![packing = num_quants];
        match comptime![layout_config.matrix_layout] {
            MatrixLayout::RowMajor => shape_col *= num_quants,
            MatrixLayout::ColMajor => shape_row *= num_quants,
        };
    }

    let shape = (shape_row, shape_col);

    // Noop for normal inputs because batch offset is cached, quantized uses logical batches
    let batch_layout = match comptime![arg.clone()] {
        MatmulArg::Normal(_) => VirtualLayout::new::<NoopLayout>(NoopLayout::new()),
        MatmulArg::Quantized { data, .. } => {
            let data_arg = comptime![MatmulArg::Normal(data)];
            input_batch_layout(inputs, batch_shape, data_arg, comptime![config.clone()])
        }
    };

    let data_layout = global_layout(
        inputs,
        shape,
        batch_layout,
        comptime![arg.data().clone()],
        comptime![config.clone()],
        data_tensor.tensor.line_size(),
        layout_config,
        packing,
    );
    let data_buf = GlobalInput::new(inputs, locals, data, comptime![config.clone()], None);

    match comptime![arg.clone()] {
        MatmulArg::Normal(_) => View::new::<GlobalInput, Coords1d>(&data_buf, data_layout),
        MatmulArg::Quantized { scales, scheme, .. } => {
            let scales_layout = match comptime![scheme.level] {
                QuantLevel::Tensor => GlobalScaleLayout::new_PerTensor(shape),
                QuantLevel::Block(block_size) => {
                    let block_size = comptime![block_size.as_dim::<2>()];

                    let scales_arg = comptime![MatmulArg::Normal(scales.clone())];
                    let batch_layout = input_batch_layout(
                        inputs,
                        batch_shape,
                        scales_arg,
                        comptime![config.clone()],
                    );

                    let scales_layout = global_layout(
                        inputs,
                        shape,
                        batch_layout,
                        comptime![scales.clone()],
                        comptime![config.clone()],
                        1u32,
                        layout_config,
                        1u32,
                    );
                    GlobalScaleLayout::new_BlockScaled(BlockScaledLayout::new(
                        shape,
                        scales_layout,
                        comptime![(block_size[0] as u32, block_size[1] as u32)],
                    ))
                }
            };
            let scales_buf = GlobalInput::new(inputs, locals, scales, config, None);
            create_quant_view_dynamic(data_buf, data_layout, scales_buf, scales_layout, scheme)
        }
    }
}

#[cube]
fn input_batch_layout(
    inputs: &GlobalArgs,
    batch_shape: &Sequence<FastDivmod>,
    #[comptime] arg: MatmulArg,
    #[comptime] config: FuseBlockConfig,
) -> VirtualLayout<u32, u32> {
    let rank = comptime![config.rank];
    match comptime![arg.clone()] {
        MatmulArg::Normal(arg) => {
            let data_tensor = match comptime![arg.clone()] {
                FuseArg::Input(pos, ..) => inputs.tensors.index(pos),
                _ => panic!("Input must be concrete"),
            };

            let mut batch_strides = Sequence::new();
            #[unroll]
            for i in 0..rank - 2 {
                let shape = data_tensor.tensor.shape(i);
                let stride = select(shape == 1, 0, data_tensor.tensor.stride(i));
                batch_strides.push(stride);
            }

            VirtualLayout::new::<BatchLayout>(BatchLayout::new(batch_strides, batch_shape.clone()))
        }
        MatmulArg::Quantized { .. } => VirtualLayout::new::<NoopLayout>(NoopLayout::new()),
    }
}

#[cube]
fn global_layout(
    inputs: &GlobalArgs,
    shape: Coords2d,
    batch_layout: VirtualLayout<u32, u32>,
    #[comptime] arg: FuseArg,
    #[comptime] config: FuseBlockConfig,
    #[comptime] line_size: u32,
    #[comptime] layout_config: GlobalLayoutConfig,
    #[comptime] packing: u32,
) -> GlobalLayout {
    let rank = comptime![config.rank];
    let data_tensor = match comptime![arg.clone()] {
        FuseArg::Input(pos, ..) => inputs.tensors.index(pos),
        _ => panic!("Input must be concrete"),
    };

    let (shape_row, shape_col) = shape;

    let stride_row = data_tensor.tensor.stride(rank - 2);
    let stride_col = data_tensor.tensor.stride(rank - 1);

    GlobalLayout::new(
        batch_layout,
        shape_row,
        shape_col,
        stride_row,
        stride_col,
        line_size,
        packing,
        layout_config,
    )
}

struct CreateQuantView<'a, E: Numeric> {
    scope: &'a mut Scope,
    data_buf: GlobalInputExpand,
    data_layout: GlobalLayoutExpand,
    scales_buf: GlobalInputExpand,
    scales_layout: GlobalScaleLayoutExpand,
    scheme: QuantScheme,
    _ty: PhantomData<E>,
}

impl<'a, E: Numeric> RunWithQuantType for CreateQuantView<'a, E> {
    type Output = ViewExpand<Line<E>, Coords3d>;

    fn execute<Q: CubePrimitive, S: CubePrimitive>(self) -> Self::Output {
        create_quant_view::expand::<E, Q, S>(
            self.scope,
            self.data_buf,
            self.data_layout,
            self.scales_buf,
            self.scales_layout,
            self.scheme,
        )
    }
}

#[cube]
#[allow(unused)]
fn create_quant_view_dynamic<E: Numeric>(
    data_buf: GlobalInput,
    data_layout: GlobalLayout,
    scales_buf: GlobalInput,
    scales_layout: GlobalScaleLayout,
    #[comptime] scheme: QuantScheme,
) -> View<Line<E>, Coords3d> {
    intrinsic!(|scope| {
        let func = CreateQuantView {
            scope,
            data_buf,
            data_layout,
            scales_buf,
            scales_layout,
            scheme,
            _ty: PhantomData,
        };
        run_with_quant_type(func, scheme)
    })
}

#[cube]
fn create_quant_view<E: Numeric, Q: CubePrimitive, S: CubePrimitive>(
    data_buf: GlobalInput,
    data_layout: GlobalLayout,
    scales_buf: GlobalInput,
    scales_layout: GlobalScaleLayout,
    #[comptime] scheme: QuantScheme,
) -> View<Line<E>, Coords3d> {
    let data_view: View<Line<Q>, Coords3d> =
        View::new::<GlobalInput, Coords1d>(&data_buf, data_layout);
    let scales_view: View<S, Coords3d> =
        View::new::<GlobalInput, Coords1d>(&scales_buf, scales_layout);
    QuantizedView::new(data_view, scales_view, scheme).view()
}

#[derive(CubeType)]
pub struct FusedMatmulState {
    inputs: GlobalArgs,
    outputs: GlobalArgs,
    locals: LocalArgs,
    a_batch: VirtualLayout<Coords1d, Coords1d>,
    b_batch: VirtualLayout<Coords1d, Coords1d>,
    c_batch: CubeOption<VirtualLayout<Coords1d, Coords1d>>,
    out_batch: VirtualLayout<Coords1d, Coords1d>,
    #[cube(comptime)]
    config: FuseBlockConfig,
    #[cube(comptime)]
    a: MatmulArg,
    #[cube(comptime)]
    b: MatmulArg,
    #[cube(comptime)]
    c: Option<MatmulArg>,
    #[cube(comptime)]
    out: FuseArg,
    #[cube(comptime)]
    lhs_layout_config: GlobalLayoutConfig,
    #[cube(comptime)]
    rhs_layout_config: GlobalLayoutConfig,
    #[cube(comptime)]
    out_layout_config: GlobalLayoutConfig,
    batch_shape: Sequence<FastDivmod>,
}

#[cube]
impl FusedMatmulState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inputs: &FusedMatmulInput,
        outputs: &mut GlobalArgs,
        locals: &mut LocalArgs,
        a_batch: VirtualLayout<u32, u32>,
        b_batch: VirtualLayout<u32, u32>,
        c_batch: CubeOption<VirtualLayout<u32, u32>>,
        out_batch: VirtualLayout<u32, u32>,
        batch_shape: Sequence<FastDivmod>,
        #[comptime] config: &FuseBlockConfig,
        #[comptime] lhs_layout_config: GlobalLayoutConfig,
        #[comptime] rhs_layout_config: GlobalLayoutConfig,
        #[comptime] out_layout_config: GlobalLayoutConfig,
    ) -> FusedMatmulState {
        FusedMatmulState {
            inputs: inputs.global.clone(),
            outputs: outputs.clone(),
            config: comptime![config.clone()],
            locals: locals.clone(),
            a_batch,
            b_batch,
            c_batch,
            out_batch,
            a: comptime![inputs.a.clone()],
            b: comptime![inputs.b.clone()],
            c: comptime![inputs.c.clone()],
            out: comptime![inputs.out.clone()],
            lhs_layout_config,
            rhs_layout_config,
            out_layout_config,
            batch_shape,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
/// Argument to a matmul operation.
pub enum MatmulArg {
    Normal(FuseArg),
    Quantized {
        data: FuseArg,
        scales: FuseArg,
        precision: FuseType,
        scheme: QuantScheme,
    },
}

impl MatmulArg {
    pub fn data(&self) -> &FuseArg {
        match self {
            MatmulArg::Normal(arg) => arg,
            MatmulArg::Quantized { data, .. } => data,
        }
    }

    pub fn scheme(&self) -> Option<&QuantScheme> {
        match self {
            MatmulArg::Normal(_) => None,
            MatmulArg::Quantized { scheme, .. } => Some(scheme),
        }
    }

    pub fn precision(&self) -> FuseType {
        match self {
            MatmulArg::Normal(arg) => arg.precision(),
            MatmulArg::Quantized { precision, .. } => *precision,
        }
    }
}
