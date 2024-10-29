use crate::{
    fusion::{on_write::ir::LayoutInfo, strides_dyn_rank, JitFusionHandle},
    JitRuntime,
};

use super::ir::{Arg, ElemwiseConfig, ElemwiseOp, ElemwisePrecision, GlobalArgsLaunch};
use burn_fusion::stream::Context;
use burn_tensor::{
    repr::{TensorDescription, TensorId, TensorStatus},
    DType,
};
use cubecl::{ir::Elem, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(new, Clone, Serialize, Deserialize)]
/// Trace containing all element wise operations as well as reads and writes.
pub struct FuseOnWriteTrace {
    outputs: RegisteredTensors,
    inputs: RegisteredTensors,
    scalars: BTreeMap<ElemwisePrecision, u32>,
    ops: Vec<ElemwiseOp>,
    reads: BTreeMap<TensorId, ElemwiseOp>,
    writes: BTreeMap<TensorId, ElemwiseOp>,
}

/// A trace runner is responsible for determining the vectorization factor as well as launching
/// a kernel based on global [inputs](GlobalArgsLaunch) and [outputs](GlobalArgsLaunch)
/// with a provided [element wise config](ElemwiseConfig).
pub trait TraceRunner<R: JitRuntime> {
    /// Run the trace.
    fn run<'a>(
        client: &ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: ElemwiseConfig,
    );
    /// The vectorization factor for all inputs and outputs.
    fn vectorization<'a>(
        handles_inputs: impl Iterator<Item = &'a JitFusionHandle<R>>,
        inputs: impl Iterator<Item = &'a TensorDescription>,
        outputs: impl Iterator<Item = &'a TensorDescription>,
    ) -> u8;
}

struct LaunchAnalysis<'a, 'c, R: JitRuntime> {
    potential_inplaces: Vec<PotentialInplace<'a>>,
    global_inputs: Vec<&'c TensorDescription>,
    global_outputs: Vec<&'c TensorDescription>,
    handle_inputs: Vec<HandleInput<'c, R>>,
    handle_outputs: Vec<HandleOutput<'c, R>>,
    reference: Option<Reference>,
    reads: BTreeMap<TensorId, ElemwiseOp>,
    writes: BTreeMap<TensorId, ElemwiseOp>,
    rank: usize,
    vectorization: u8,
}

#[derive(Debug)]
enum HandleOutput<'c, R: JitRuntime> {
    Alias {
        input_pos: usize,
        precision: ElemwisePrecision,
    },
    Owned {
        precision: ElemwisePrecision,
        handle: JitFusionHandle<R>,
        global_shape: &'c [usize],
    },
}

struct HandleInput<'c, R: JitRuntime> {
    relative_id: TensorId,
    precision: ElemwisePrecision,
    handle: JitFusionHandle<R>,
    global_shape: &'c [usize],
}

struct Reference {
    layout: Arg,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

struct PotentialInplace<'a> {
    input_pos: usize,
    tensor_relative: &'a TensorDescription,
    strides: Vec<usize>,
}

impl FuseOnWriteTrace {
    /// Run a trace with the given [runner](TraceRunner).
    pub fn run<R: JitRuntime, Runner: TraceRunner<R>>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, JitFusionHandle<R>>,
    ) {
        let analysis = self.analyse::<R, Runner>(client, device, context);

        let inputs = self.register_inputs(context, &analysis.handle_inputs, analysis.vectorization);
        let outputs = self.register_outputs(&analysis.handle_outputs, analysis.vectorization);

        let mut ops = Sequence::new();
        for op in analysis.reads.into_values() {
            ops.push(op);
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

        Runner::run(client, inputs, outputs, config)
    }

    fn analyse<'a, 'c, R: JitRuntime, Runner: TraceRunner<R>>(
        &'a self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'c, JitFusionHandle<R>>,
    ) -> LaunchAnalysis<'a, 'c, R> {
        let mut analysis = LaunchAnalysis {
            potential_inplaces: Vec::new(),
            global_inputs: Vec::new(),
            global_outputs: Vec::new(),
            handle_inputs: Vec::new(),
            handle_outputs: Vec::new(),
            reference: None,
            reads: self.reads.clone(),
            writes: self.writes.clone(),
            rank: 1,
            vectorization: 1,
        };

        self.analyse_inputs(context, &mut analysis);
        self.analyse_outputs(client, device, context, &mut analysis);

        analysis.vectorization = Runner::vectorization(
            analysis.handle_inputs.iter().map(|item| &item.handle),
            analysis.global_inputs.iter().copied(),
            analysis.global_outputs.iter().copied(),
        );

        analysis
    }

    fn analyse_inputs<'a, 'c, R: JitRuntime>(
        &'a self,
        context: &mut Context<'c, JitFusionHandle<R>>,
        analysis: &mut LaunchAnalysis<'a, 'c, R>,
    ) {
        for (i, (precision, tensor_relative)) in self.inputs.iter().enumerate() {
            let tensor_global = context.tensors.get(&tensor_relative.id).unwrap();
            // Important to take the status of the relative graph and not
            // the global graph, since the status of the global graph
            // might be of a later operation on the same tensor id.
            let status = &tensor_relative.status;
            let handle = context.handles.get_handle(&tensor_global.id, status);

            if status == &TensorStatus::ReadWrite && handle.handle.can_mut() {
                analysis.potential_inplaces.push(PotentialInplace {
                    input_pos: i,
                    tensor_relative,
                    strides: handle.strides.clone(),
                });
            }

            analysis.global_inputs.push(tensor_global);
            analysis.rank = usize::max(tensor_global.shape.len(), analysis.rank);
            analysis.handle_inputs.push(HandleInput {
                precision,
                handle,
                relative_id: tensor_relative.id,
                global_shape: &tensor_global.shape,
            });
        }
    }

    fn analyse_outputs<'a, 'c, R: JitRuntime>(
        &'a self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'c, JitFusionHandle<R>>,
        analysis: &mut LaunchAnalysis<'a, 'c, R>,
    ) {
        for (precision, tensor_relative) in self.outputs.iter() {
            let tensor_global = context.tensors.get(&tensor_relative.id).unwrap();
            let strides = strides_dyn_rank(&tensor_global.shape);
            analysis.global_outputs.push(tensor_global);

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

                    if let ElemwiseOp::Assign(op) =
                        analysis.reads.get_mut(&handle_input.relative_id).unwrap()
                    {
                        op.input.add_layout_info(LayoutInfo::IsRef);
                    };

                    if let ElemwiseOp::Assign(op) =
                        analysis.writes.get_mut(&tensor_relative.id).unwrap()
                    {
                        op.out.add_layout_info(LayoutInfo::IsRef);
                    };
                }

                context
                    .handles
                    .register_handle(tensor_global.id, handle_input.handle.clone());
                analysis.handle_outputs.push(HandleOutput::Alias {
                    input_pos: potential_inplace.input_pos,
                    precision,
                });
            } else {
                if analysis.reference.is_none() {
                    analysis.reference = Some(Reference {
                        layout: Arg::Output(0, precision, LayoutInfo::IsRef),
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

                // We encode bool tensors as u32.
                let dtype = match tensor_global.dtype {
                    DType::Bool => DType::U32,
                    _ => tensor_global.dtype,
                };
                let size = tensor_global.shape.iter().product::<usize>() * Elem::from(dtype).size();

                let handle = JitFusionHandle {
                    client: client.clone(),
                    handle: client.empty(size),
                    device: device.clone(),
                    strides,
                };

                analysis.rank = usize::max(tensor_global.shape.len(), analysis.rank);
                context
                    .handles
                    .register_handle(tensor_global.id, handle.clone());

                analysis.handle_outputs.push(HandleOutput::Owned {
                    precision,
                    handle,
                    global_shape: &tensor_global.shape,
                });
            }
        }

        Self::add_layout_info_inputs(analysis);
    }

    fn add_layout_info_inputs<R: JitRuntime>(analysis: &mut LaunchAnalysis<'_, '_, R>) {
        for hi in analysis.handle_inputs.iter() {
            if let Some(reference) = analysis.reference.as_ref() {
                if reference.strides == hi.handle.strides && reference.shape == hi.global_shape {
                    if let ElemwiseOp::Assign(op) = analysis.reads.get_mut(&hi.relative_id).unwrap()
                    {
                        op.input.add_layout_info(LayoutInfo::SameAsRef);
                    }
                }
            }
        }
    }

    fn register_inputs<'c, 'h, R: JitRuntime>(
        &self,
        context: &mut Context<'c, JitFusionHandle<R>>,
        handle_inputs: &'h [HandleInput<'c, R>],
        vectorization: u8,
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
        );

        for hi in handle_inputs.iter() {
            let arg = hi.handle.as_tensor_arg(hi.global_shape, vectorization);
            match hi.precision {
                ElemwisePrecision::F32 => inputs.t_f32.push(arg),
                ElemwisePrecision::F16 => inputs.t_f16.push(arg),
                ElemwisePrecision::BF16 => inputs.t_bf16.push(arg),
                ElemwisePrecision::I32 => inputs.t_i32.push(arg),
                ElemwisePrecision::U32 => inputs.t_u32.push(arg),
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
                    ElemwisePrecision::I32 => {
                        inputs.s_i32.push(ScalarArg::new(context.scalar_ints[i]))
                    }
                    _ => todo!(),
                }
            }
        }

        inputs
    }

    fn register_outputs<'s, R: JitRuntime>(
        &self,
        handle_outputs: &'s [HandleOutput<'_, R>],
        vectorization: u8,
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
        );
        for item in handle_outputs.iter() {
            match item {
                HandleOutput::Alias {
                    input_pos,
                    precision,
                } => match precision {
                    ElemwisePrecision::F32 => outputs.t_f32.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::F16 => outputs.t_f16.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::I32 => outputs.t_i32.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::U32 => outputs.t_u32.push(TensorArg::alias(*input_pos)),
                    _ => todo!(),
                },
                HandleOutput::Owned {
                    precision,
                    handle,
                    global_shape,
                } => {
                    let arg = handle.as_tensor_arg(global_shape, vectorization);

                    match precision {
                        ElemwisePrecision::F32 => outputs.t_f32.push(arg),
                        ElemwisePrecision::F16 => outputs.t_f16.push(arg),
                        ElemwisePrecision::I32 => outputs.t_i32.push(arg),
                        ElemwisePrecision::U32 => outputs.t_u32.push(arg),
                        // Bools are encoded as u32.
                        ElemwisePrecision::Bool => outputs.t_u32.push(arg),
                        _ => todo!(),
                    };
                }
            }
        }

        outputs
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct RegisteredTensors {
    tensors: BTreeMap<ElemwisePrecision, Vec<TensorDescription>>,
}

impl RegisteredTensors {
    pub fn iter(&self) -> impl Iterator<Item = (ElemwisePrecision, &TensorDescription)> {
        self.tensors.iter().flat_map(|(precision, descriptions)| {
            descriptions.iter().map(|desc| (*precision, desc))
        })
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
