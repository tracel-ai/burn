use crate::{
    fusion::{on_write::ir::LayoutInfo, strides_dyn_rank, JitFusionHandle},
    BoolElement, JitRuntime,
};

use super::ir::{Arg, ElemwiseConfig, ElemwiseOp, ElemwisePrecision, GlobalArgsLaunch};
use super::position::PositionMapper;
use burn_fusion::stream::Context;
use burn_tensor::{
    repr::{TensorDescription, TensorId, TensorStatus},
    DType,
};
use cubecl::{ir::Elem, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(new, Clone, Serialize, Deserialize, Debug)]
/// Trace containing all element wise operations as well as reads and writes.
pub struct FuseOnWriteTrace {
    outputs: RegisteredTensors,
    inputs: RegisteredTensors,
    scalars: BTreeMap<ElemwisePrecision, u32>,
    reshapes: Vec<Reshape>,
    shape_ref: Vec<usize>,
    ops: Vec<ElemwiseOp>,
    reads: BTreeMap<TensorId, Vec<ElemwiseOp>>,
    writes: BTreeMap<TensorId, ElemwiseOp>,
    inputs_unhandled: Vec<TensorId>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Reshape {
    pub reshaped: TensorId,
    pub original: TensorId,
}

/// A trace runner is responsible for determining the vectorization factor as well as launching
/// a kernel based on global [inputs](GlobalArgsLaunch) and [outputs](GlobalArgsLaunch)
/// with a provided [element wise config](ElemwiseConfig).
pub trait TraceRunner<R: JitRuntime> {
    /// The error that might happen while running the trace.
    type Error;

    /// Run the trace.
    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: &'a ElemwiseConfig,
    ) -> Result<(), Self::Error>;

    /// The vectorization factor for all inputs and outputs.
    fn vectorization<'a>(
        vectorizations: &mut BTreeMap<TensorId, u8>,
        handles_inputs: impl Iterator<Item = &'a JitFusionHandle<R>>,
        inputs: impl Iterator<Item = &'a TensorDescription>,
        outputs: impl Iterator<Item = &'a TensorDescription>,
        reshaped: impl Iterator<Item = (&'a TensorDescription, &'a TensorDescription, bool)>,
    ) {
        enum Vect {
            Broadcated,
            Max(u8),
        }

        // The default version uses the last dimension as vectorization axis and assumes a
        // perpendicular contiguous line.
        let vectorization_input = |handle: &JitFusionHandle<R>, desc: &TensorDescription| {
            let rank = handle.strides.len();

            // Last dimension strides should be 1, otherwise vecX won't be contiguous.
            if handle.strides[rank - 1] != 1 {
                return Vect::Max(1);
            }
            let shape_axis = desc.shape[rank - 1];

            if shape_axis == 1 {
                return Vect::Broadcated;
            }

            for s in R::line_size_elem(&desc.dtype.into()) {
                // The last dimension should be a multiple of the vector size or broadcated.
                if shape_axis % s as usize == 0 {
                    return Vect::Max(s);
                }
            }

            Vect::Max(1)
        };

        let vectorization_output = |desc: &TensorDescription| {
            let rank = desc.shape.len();

            for s in R::line_size_elem(&desc.dtype.into()) {
                // The last dimension should be a multiple of the vector size.
                if desc.shape[rank - 1] % s as usize == 0 {
                    return Vect::Max(s);
                }
            }

            Vect::Max(1)
        };

        let vectorization_reshape =
            |reshaped: &TensorDescription, original: &TensorDescription, multi_reads: bool| {
                let reshape_axis = reshaped.shape[reshaped.shape.len() - 1];
                let shape_axis = original.shape[original.shape.len() - 1];

                if !multi_reads && reshape_axis == 1 {
                    return Vect::Broadcated;
                }

                for s in R::line_size_elem(&reshaped.dtype.into()) {
                    if !multi_reads {
                        // The last dimension should be a multiple of the vector size or broadcated.
                        if reshape_axis % s as usize == 0 {
                            return Vect::Max(s);
                        }
                    } else {
                        // Since the original tensor must share the same vectorization factor as the
                        // reshaped tensor, they must have compatible shapes when both are access
                        // independently.
                        if reshape_axis % s as usize == 0 && shape_axis % s as usize == 0 {
                            return Vect::Max(s);
                        }
                    }
                }

                Vect::Max(1)
            };

        let mut max_current = u8::MAX;

        for (handle, tensor) in handles_inputs.zip(inputs) {
            match vectorization_input(&handle, tensor) {
                Vect::Broadcated => vectorizations.insert(tensor.id, 1),
                Vect::Max(val) => {
                    max_current = Ord::min(val, max_current);
                    vectorizations.insert(tensor.id, 0)
                }
            };
        }

        for tensor in outputs {
            match vectorization_output(tensor) {
                Vect::Broadcated => vectorizations.insert(tensor.id, 1),
                Vect::Max(val) => {
                    max_current = Ord::min(val, max_current);
                    vectorizations.insert(tensor.id, 0)
                }
            };
        }

        for (reshaped, original, multi_reads) in reshaped {
            match vectorization_reshape(reshaped, original, multi_reads) {
                Vect::Broadcated => {
                    vectorizations.insert(original.id, 1);
                    vectorizations.insert(reshaped.id, 1);
                }
                Vect::Max(val) => {
                    vectorizations.insert(original.id, 0);
                    vectorizations.insert(reshaped.id, 0);
                    max_current = Ord::min(val, max_current);
                }
            }
        }

        for (_id, val) in vectorizations.iter_mut() {
            if *val == 0 {
                *val = max_current;
            }
        }
    }
}

#[derive(Debug)]
struct LaunchAnalysis<'a, R: JitRuntime> {
    potential_inplaces: Vec<PotentialInplace<'a>>,
    global_inputs: Vec<TensorDescription>,
    global_outputs: Vec<TensorDescription>,
    handle_inputs: Vec<HandleInput<R>>,
    handle_outputs: Vec<HandleOutput<R>>,
    reference: Option<Reference>,
    reads: BTreeMap<TensorId, Vec<ElemwiseOp>>,
    writes: BTreeMap<TensorId, ElemwiseOp>,
    vectorization: BTreeMap<TensorId, u8>,
    rank: usize,
}

#[derive(Debug)]
enum HandleOutput<R: JitRuntime> {
    Alias {
        input_pos: usize,
        precision: ElemwisePrecision,
    },
    Owned {
        global_id: TensorId,
        precision: ElemwisePrecision,
        handle: JitFusionHandle<R>,
        global_shape: Vec<usize>,
        vectorization: u8,
    },
}

#[derive(Debug)]
struct HandleInput<R: JitRuntime> {
    relative_id: TensorId,
    global_id: TensorId,
    precision: ElemwisePrecision,
    handle: JitFusionHandle<R>,
    global_shape: Vec<usize>,
    vectorization: u8,
}

#[derive(Debug)]
struct Reference {
    layout: Arg,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

#[derive(Debug)]
struct PotentialInplace<'a> {
    input_pos: usize,
    tensor_relative: &'a TensorDescription,
    strides: Vec<usize>,
}

impl FuseOnWriteTrace {
    /// Run a trace with the given [runner](TraceRunner).
    pub fn run<R: JitRuntime, BT: BoolElement, Runner: TraceRunner<R>>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, JitFusionHandle<R>>,
        runner: &Runner,
    ) -> Result<(), Runner::Error> {
        let analysis = self.analyse::<R, BT, Runner>(client, device, context);
        let inputs = self.register_inputs(context, &analysis.handle_inputs);
        let outputs = self.register_outputs::<_, BT>(&analysis.handle_outputs);

        let mut ops = Sequence::<ElemwiseOp>::new();

        for read_ops in analysis.reads.into_values() {
            for op in read_ops {
                ops.push(op);
            }
        }

        for op in self.ops.iter() {
            ops.push(op.clone());
        }

        for op in analysis.writes.into_values() {
            ops.push(op);
        }

        let config = ElemwiseConfig {
            rank: analysis.rank as u32,
            ref_layout: analysis
                .reference
                .expect("An output should exist for the fused kernel")
                .layout,
            ops,
        };

        match Runner::run(runner, client, inputs, outputs, &config) {
            Err(err) => {
                self.rollback(context, analysis.handle_inputs, analysis.handle_outputs);
                Err(err)
            }
            Ok(val) => Ok(val),
        }
    }

    fn rollback<R: JitRuntime>(
        &self,
        context: &mut Context<'_, JitFusionHandle<R>>,
        handle_inputs: Vec<HandleInput<R>>,
        handle_outputs: Vec<HandleOutput<R>>,
    ) {
        for input in handle_inputs {
            context
                .handles
                .register_handle(input.global_id, input.handle);
        }
        for output in handle_outputs {
            if let HandleOutput::Owned {
                global_id, handle, ..
            } = output
            {
                context.handles.register_handle(global_id, handle);
            }
        }
    }

    fn analyse<'a, R: JitRuntime, BT: BoolElement, Runner: TraceRunner<R>>(
        &'a self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, JitFusionHandle<R>>,
    ) -> LaunchAnalysis<'a, R> {
        let mut analysis = LaunchAnalysis {
            potential_inplaces: Vec::new(),
            global_inputs: Vec::new(),
            global_outputs: Vec::new(),
            handle_inputs: Vec::new(),
            handle_outputs: Vec::new(),
            reference: None,
            vectorization: BTreeMap::default(),
            reads: self.reads.clone(),
            writes: self.writes.clone(),
            rank: self.shape_ref.len(),
        };

        self.analyse_inputs(context, &mut analysis);
        self.analyse_outputs::<_, BT>(client, device, context, &mut analysis);

        let tensors_reshaped = self.reshapes.iter().map(|reshape| {
            (
                context.tensors.get(&reshape.reshaped).unwrap(),
                context.tensors.get(&reshape.original).unwrap(),
                self.reads.get(&reshape.original).unwrap().len() > 1,
            )
        });

        Runner::vectorization(
            &mut analysis.vectorization,
            analysis.handle_inputs.iter().map(|item| &item.handle),
            analysis.global_inputs.iter(),
            analysis.global_outputs.iter(),
            tensors_reshaped,
        );

        for handle in analysis.handle_inputs.iter_mut() {
            handle.vectorization = *analysis.vectorization.get(&handle.global_id).unwrap();
        }
        for handle in analysis.handle_outputs.iter_mut() {
            match handle {
                HandleOutput::Owned {
                    vectorization,
                    global_id,
                    ..
                } => *vectorization = *analysis.vectorization.get(&global_id).unwrap(),
                _ => {}
            }
        }

        analysis
    }

    fn analyse_inputs<'a, R: JitRuntime>(
        &'a self,
        context: &mut Context<'_, JitFusionHandle<R>>,
        analysis: &mut LaunchAnalysis<'a, R>,
    ) {
        for (i, (precision, tensor_relative)) in self.inputs.iter().enumerate() {
            let mut tensor_global = context.tensors.get(&tensor_relative.id).unwrap().clone();
            // Important to take the status of the relative graph and not
            // the global graph, since the status of the global graph
            // might be of a later operation on the same tensor id.
            let status = &tensor_relative.status;
            let mut handle = context.handles.get_handle(&tensor_global.id, status);

            if status == &TensorStatus::ReadWrite
                && handle.handle.can_mut()
                && !self.inputs_unhandled.contains(&tensor_relative.id)
                && self
                    .reshapes
                    .iter()
                    .find(|r| r.reshaped == tensor_relative.id || r.original == tensor_relative.id)
                    .is_none()
                && self.shape_ref == tensor_relative.shape
            {
                analysis.potential_inplaces.push(PotentialInplace {
                    input_pos: i,
                    tensor_relative,
                    strides: handle.strides.clone(),
                });
            }

            if tensor_global.shape.len() < analysis.rank {
                let num_elem: usize = tensor_global.shape.iter().product();
                for _ in 0..(analysis.rank - tensor_global.shape.len()) {
                    tensor_global.shape.insert(0, 1);
                    handle.strides.insert(0, num_elem);
                }
            }

            analysis.handle_inputs.push(HandleInput {
                precision,
                handle,
                relative_id: tensor_relative.id,
                global_id: tensor_global.id,
                global_shape: tensor_global.shape.clone(),
                vectorization: 1,
            });
            analysis.global_inputs.push(tensor_global);
        }
    }

    fn analyse_outputs<'a, R: JitRuntime, BT: BoolElement>(
        &'a self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, JitFusionHandle<R>>,
        analysis: &mut LaunchAnalysis<'a, R>,
    ) {
        let mut position_mapper = PositionMapper::default();
        let mut output_sorted: Vec<_> = self
            .outputs
            .iter()
            .enumerate()
            .map(|(pos, (precision, tensor))| {
                position_mapper.register(precision, pos);
                (pos, (precision, tensor))
            })
            .collect();

        output_sorted.sort_by(|(_, (_, a)), (_, (_, b))| {
            let a_val: usize = a.shape.iter().sum();
            let b_val: usize = b.shape.iter().sum();

            b_val.cmp(&a_val)
        });
        let mut handles = Vec::with_capacity(self.outputs.len());
        let mut globals = Vec::with_capacity(self.outputs.len());

        for _ in 0..self.outputs.len() {
            handles.push(None);
            globals.push(None);
        }

        for (position_original, (precision, tensor_relative)) in output_sorted.into_iter() {
            let tensor_global = context.tensors.get(&tensor_relative.id).unwrap().clone();
            let strides = strides_dyn_rank(&tensor_global.shape);

            if let Some(index) = analysis
                .potential_inplaces
                .iter()
                .enumerate()
                .find(|(_pos, pi)| {
                    pi.tensor_relative.dtype == tensor_global.dtype
                        && pi.tensor_relative.shape == tensor_relative.shape
                        && pi.strides == strides
                })
                .map(|(pos, _)| pos)
            {
                let potential_inplace = analysis.potential_inplaces.remove(index);
                let handle_input = analysis
                    .handle_inputs
                    .get(potential_inplace.input_pos)
                    .unwrap();

                if analysis.reference.is_none() {
                    let index_input = self
                        .inputs
                        .get_index(precision, potential_inplace.tensor_relative.id)
                        .unwrap();

                    analysis.reference = Some(Reference {
                        layout: Arg::Input(index_input as u32, precision, LayoutInfo::IsRef),
                        shape: tensor_global.shape.clone(),
                        strides: handle_input.handle.strides.clone(),
                    });

                    if let Some(ops) = analysis.reads.get_mut(&handle_input.relative_id) {
                        for op in ops.iter_mut() {
                            if let ElemwiseOp::Assign(op) = op {
                                op.input.add_layout_info(LayoutInfo::IsRef);
                            };
                        }
                    }

                    if let Some(ElemwiseOp::Assign(op)) =
                        analysis.writes.get_mut(&tensor_relative.id)
                    {
                        op.out.add_layout_info(LayoutInfo::IsRef);
                    };
                }

                context
                    .handles
                    .register_handle(tensor_global.id, handle_input.handle.clone());

                handles[position_original] = Some(HandleOutput::Alias {
                    input_pos: potential_inplace.input_pos,
                    precision,
                });
                globals[position_original] = Some(tensor_global);
            } else {
                if analysis.reference.is_none() {
                    let position = position_mapper.resolve_index(&precision, position_original);
                    analysis.reference = Some(Reference {
                        layout: Arg::Output(position, precision, LayoutInfo::IsRef),
                        shape: tensor_global.shape.clone(),
                        strides: strides.clone(),
                    });

                    if let ElemwiseOp::Assign(op) =
                        analysis.writes.get_mut(&tensor_relative.id).unwrap()
                    {
                        op.out.add_layout_info(LayoutInfo::IsRef);
                    };
                } else if let Some(reference) = analysis.reference.as_ref() {
                    if reference.strides == strides && reference.shape == tensor_global.shape {
                        if let ElemwiseOp::Assign(op) =
                            analysis.writes.get_mut(&tensor_relative.id).unwrap()
                        {
                            op.out.add_layout_info(LayoutInfo::SameAsRef);
                        };
                    }
                }

                // We encode bool tensors as `B`.
                let dtype = match tensor_global.dtype {
                    DType::Bool => BT::dtype(),
                    _ => tensor_global.dtype,
                };
                let size = tensor_global.shape.iter().product::<usize>() * Elem::from(dtype).size();

                let handle = JitFusionHandle {
                    client: client.clone(),
                    handle: client.empty(size),
                    device: device.clone(),
                    strides,
                    dtype,
                };

                analysis.rank = usize::max(tensor_global.shape.len(), analysis.rank);
                context
                    .handles
                    .register_handle(tensor_global.id, handle.clone());

                handles[position_original] = Some(HandleOutput::Owned {
                    precision,
                    handle,
                    global_shape: tensor_global.shape.clone(),
                    global_id: tensor_global.id,
                    vectorization: 1,
                });
                globals[position_original] = Some(tensor_global);
            }
        }

        for (handle, global) in handles.into_iter().zip(globals.into_iter()) {
            analysis.handle_outputs.push(handle.unwrap());
            analysis.global_outputs.push(global.unwrap());
        }

        Self::add_layout_info_inputs(analysis);
    }

    fn add_layout_info_inputs<R: JitRuntime>(analysis: &mut LaunchAnalysis<'_, R>) {
        for hi in analysis.handle_inputs.iter() {
            if let Some(reference) = analysis.reference.as_ref() {
                if reference.strides == hi.handle.strides && reference.shape == hi.global_shape {
                    if let Some(ops) = analysis.reads.get_mut(&hi.relative_id) {
                        for op in ops.iter_mut() {
                            if let ElemwiseOp::Assign(op) = op {
                                op.input.add_layout_info(LayoutInfo::SameAsRef);
                            }
                        }
                    }
                }
            }
        }
    }

    fn register_inputs<'h, R: JitRuntime>(
        &self,
        context: &mut Context<'_, JitFusionHandle<R>>,
        handle_inputs: &'h [HandleInput<R>],
    ) -> GlobalArgsLaunch<'h, R> {
        let mut inputs = GlobalArgsLaunch::new(
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
        );

        for hi in handle_inputs.iter() {
            let arg = hi.handle.as_tensor_arg(&hi.global_shape, hi.vectorization);
            match hi.precision {
                ElemwisePrecision::F32 => inputs.t_f32.push(arg),
                ElemwisePrecision::F16 => inputs.t_f16.push(arg),
                ElemwisePrecision::BF16 => inputs.t_bf16.push(arg),
                ElemwisePrecision::I64 => inputs.t_i64.push(arg),
                ElemwisePrecision::I32 => inputs.t_i32.push(arg),
                ElemwisePrecision::I16 => inputs.t_i16.push(arg),
                ElemwisePrecision::I8 => inputs.t_i8.push(arg),
                ElemwisePrecision::U64 => inputs.t_u64.push(arg),
                ElemwisePrecision::U32 => inputs.t_u32.push(arg),
                ElemwisePrecision::U16 => inputs.t_u16.push(arg),
                ElemwisePrecision::U8 => inputs.t_u8.push(arg),
                _ => panic!("Unsupported input precision {:?}", hi.precision),
            };
        }

        for (precision, count) in self.scalars.iter() {
            for i in 0..(*count as usize) {
                match precision {
                    ElemwisePrecision::F32 => {
                        inputs.s_f32.push(ScalarArg::new(context.scalar_f32[i]))
                    }
                    ElemwisePrecision::F16 => {
                        inputs.s_f16.push(ScalarArg::new(context.scalar_f16[i]))
                    }
                    ElemwisePrecision::BF16 => {
                        inputs.s_bf16.push(ScalarArg::new(context.scalar_bf16[i]))
                    }
                    ElemwisePrecision::I64 => {
                        inputs.s_i64.push(ScalarArg::new(context.scalar_i64[i]))
                    }
                    ElemwisePrecision::I32 => {
                        inputs.s_i32.push(ScalarArg::new(context.scalar_i32[i]))
                    }
                    ElemwisePrecision::I16 => {
                        inputs.s_i16.push(ScalarArg::new(context.scalar_i16[i]))
                    }
                    ElemwisePrecision::I8 => inputs.s_i8.push(ScalarArg::new(context.scalar_i8[i])),
                    ElemwisePrecision::U64 => {
                        inputs.s_u64.push(ScalarArg::new(context.scalar_u64[i]))
                    }
                    ElemwisePrecision::U32 => {
                        inputs.s_u32.push(ScalarArg::new(context.scalar_u32[i]))
                    }
                    ElemwisePrecision::U16 => {
                        inputs.s_u16.push(ScalarArg::new(context.scalar_u16[i]))
                    }
                    ElemwisePrecision::U8 => inputs.s_u8.push(ScalarArg::new(context.scalar_u8[i])),
                    ElemwisePrecision::Bool => todo!(),
                }
            }
        }

        for relative in self.reshapes.iter().rev() {
            let global = context.tensors.get(&relative.reshaped).unwrap();

            for shape in global.shape.iter().rev() {
                inputs.s_u32.push(ScalarArg::new(*shape as u32))
            }
        }

        inputs
    }

    fn register_outputs<'s, R: JitRuntime, BT: BoolElement>(
        &self,
        handle_outputs: &'s [HandleOutput<R>],
    ) -> GlobalArgsLaunch<'s, R> {
        let mut outputs = GlobalArgsLaunch::new(
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
        );
        for item in handle_outputs.iter() {
            match item {
                HandleOutput::Alias {
                    input_pos,
                    precision,
                } => match precision {
                    ElemwisePrecision::F32 => outputs.t_f32.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::F16 => outputs.t_f16.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::BF16 => outputs.t_bf16.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::I64 => outputs.t_i64.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::I32 => outputs.t_i32.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::I16 => outputs.t_i16.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::I8 => outputs.t_i8.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::U64 => outputs.t_u64.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::U32 => outputs.t_u32.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::U16 => outputs.t_u16.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::U8 => outputs.t_u8.push(TensorArg::alias(*input_pos)),
                    _ => todo!(),
                },
                HandleOutput::Owned {
                    precision,
                    handle,
                    global_shape,
                    vectorization,
                    ..
                } => {
                    let arg = handle.as_tensor_arg(global_shape, *vectorization);

                    match precision {
                        ElemwisePrecision::F32 => outputs.t_f32.push(arg),
                        ElemwisePrecision::F16 => outputs.t_f16.push(arg),
                        ElemwisePrecision::BF16 => outputs.t_bf16.push(arg),
                        ElemwisePrecision::I64 => outputs.t_i64.push(arg),
                        ElemwisePrecision::I32 => outputs.t_i32.push(arg),
                        ElemwisePrecision::I16 => outputs.t_i16.push(arg),
                        ElemwisePrecision::I8 => outputs.t_i8.push(arg),
                        ElemwisePrecision::U64 => outputs.t_u64.push(arg),
                        ElemwisePrecision::U32 => outputs.t_u32.push(arg),
                        ElemwisePrecision::U16 => outputs.t_u16.push(arg),
                        ElemwisePrecision::U8 => outputs.t_u8.push(arg),
                        ElemwisePrecision::Bool => match BT::dtype() {
                            DType::U32 => outputs.t_u32.push(arg),
                            DType::U8 => outputs.t_u8.push(arg),
                            _ => todo!(),
                        },
                    };
                }
            }
        }

        outputs
    }
}

#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub struct RegisteredTensors {
    tensors: BTreeMap<ElemwisePrecision, Vec<TensorDescription>>,
}

impl RegisteredTensors {
    pub fn iter(&self) -> impl Iterator<Item = (ElemwisePrecision, &TensorDescription)> {
        self.tensors.iter().flat_map(|(precision, descriptions)| {
            descriptions.iter().map(|desc| (*precision, desc))
        })
    }

    pub fn len(&self) -> usize {
        self.tensors.values().map(|v| v.len()).sum()
    }

    pub fn get_index(&self, precision: ElemwisePrecision, tensor_id: TensorId) -> Option<usize> {
        self.tensors.get(&precision).and_then(|items| {
            items
                .iter()
                .enumerate()
                .find(|(_pos, tensor)| tensor.id == tensor_id)
                .map(|(pos, _)| pos)
        })
    }

    pub fn get_all(&self, precision: ElemwisePrecision) -> &[TensorDescription] {
        self.tensors
            .get(&precision)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn get(
        &self,
        precision: ElemwisePrecision,
        tensor_id: TensorId,
    ) -> Option<&TensorDescription> {
        self.get_all(precision)
            .iter()
            .find(|desc| desc.id == tensor_id)
    }

    pub fn insert(&mut self, precision: ElemwisePrecision, tensor: TensorDescription) -> u32 {
        if let Some(tensors) = self.tensors.get_mut(&precision) {
            let position = tensors.len() as u32;
            tensors.push(tensor);
            position
        } else {
            self.tensors.insert(precision, vec![tensor]);
            0
        }
    }

    pub fn update(&mut self, precision: ElemwisePrecision, tensor: &TensorDescription) {
        if let Some(tensors) = self.tensors.get_mut(&precision) {
            if let Some(tensor_old) = tensors
                .iter_mut()
                .find(|tensor_old| tensor_old.id == tensor.id)
            {
                tensor_old.status = tensor.status.clone();
            }
        }
    }
}
