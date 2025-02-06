use burn_fusion::stream::Context;
use burn_tensor::{repr::TensorDescription, DType};
use cubecl::{client::ComputeClient, ir::Elem};

use crate::{
    fusion::{
        on_write::ir::{Arg, ElemwiseOp, LayoutInfo},
        strides_dyn_rank, JitFusionHandle,
    },
    tensor::is_contiguous,
    BoolElement, JitRuntime,
};

use super::{
    super::ir::ElemwisePrecision, HandleOutput, LaunchPlan, Reference, RegisteredTensors, Reshape,
};
use std::collections::BTreeMap;

/// Create or reuse handles for the outputs.
///
/// It is also responsible to select the reference tensor.
pub struct OutputPlanner<'a, R: JitRuntime> {
    inputs: &'a RegisteredTensors,
    reshapes: &'a Vec<Reshape>,
    outputs_sorted: Vec<OutputSorted<'a>>,
    handles: Vec<Option<HandleOutput<R>>>,
    globals: Vec<Option<TensorDescription>>,
    mapper: OutputPositionMapper,
}

struct OutputSorted<'a> {
    pos_original: usize,
    precision: ElemwisePrecision,
    tensor_relative: &'a TensorDescription,
}

enum OutputKind {
    Normal,
    Inplace { input_pos: usize },
    Reshaped { reshape: Reshape },
}

impl<'a, R: JitRuntime> OutputPlanner<'a, R> {
    pub fn new(
        inputs: &'a RegisteredTensors,
        outputs: &'a RegisteredTensors,
        reshapes: &'a Vec<Reshape>,
    ) -> Self {
        let mut mapper = OutputPositionMapper::default();
        let mut outputs_sorted: Vec<_> = outputs
            .iter()
            .enumerate()
            .map(|(pos, (precision, tensor))| {
                mapper.register(precision, pos);
                OutputSorted {
                    pos_original: pos,
                    precision,
                    tensor_relative: tensor,
                }
            })
            .collect();

        outputs_sorted.sort_by(|a, b| {
            let a_val: usize = a.tensor_relative.shape.iter().sum();
            let b_val: usize = b.tensor_relative.shape.iter().sum();

            b_val.cmp(&a_val)
        });

        let mut handles = Vec::with_capacity(outputs.len());
        let mut globals = Vec::with_capacity(outputs.len());

        for _ in 0..outputs.len() {
            handles.push(None);
            globals.push(None);
        }

        Self {
            inputs,
            outputs_sorted,
            reshapes,
            handles,
            globals,
            mapper,
        }
    }

    pub fn run<BT: BoolElement>(
        mut self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, JitFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
    ) {
        // So that we can borrow self during the iteration.
        let mut outputs = Vec::new();
        core::mem::swap(&mut outputs, &mut self.outputs_sorted);

        for output in outputs.into_iter() {
            let tensor_global = context
                .tensors
                .get(&output.tensor_relative.id)
                .unwrap()
                .clone();
            let strides = strides_dyn_rank(&tensor_global.shape);

            match self.output_kind(plan, &tensor_global, &output, &strides) {
                OutputKind::Inplace { input_pos } => {
                    self.inplace_output(context, plan, output, tensor_global, input_pos);
                }
                OutputKind::Normal => {
                    self.normal_output::<BT>(
                        client,
                        device,
                        context,
                        plan,
                        output,
                        tensor_global,
                        strides,
                    );
                }
                OutputKind::Reshaped { reshape } => {
                    self.reshaped_output::<BT>(
                        client,
                        device,
                        context,
                        plan,
                        output,
                        tensor_global,
                        strides,
                        reshape,
                    );
                }
            }
        }

        for (handle, global) in self.handles.into_iter().zip(self.globals.into_iter()) {
            plan.handle_outputs.push(handle.unwrap());
            plan.global_outputs.push(global.unwrap());
        }

        Self::add_layout_info_inputs(plan);
    }

    fn add_layout_info_inputs(analysis: &mut LaunchPlan<'_, R>) {
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

    fn output_kind(
        &self,
        plan: &mut LaunchPlan<'a, R>,
        tensor_global: &TensorDescription,
        output: &OutputSorted,
        strides: &[usize],
    ) -> OutputKind {
        if let Some(reshape) = self
            .reshapes
            .iter()
            .find(|r| r.reshaped == output.tensor_relative.id)
        {
            return OutputKind::Reshaped {
                reshape: reshape.clone(),
            };
        }

        plan.potential_inplaces
            .iter()
            .enumerate()
            .find(|(_pos, pi)| {
                pi.tensor_relative.dtype == tensor_global.dtype
                    && pi.tensor_relative.shape == output.tensor_relative.shape
                    && pi.strides == strides
            })
            .map(|(pos, _)| pos)
            .map(|input_pos| OutputKind::Inplace { input_pos })
            .unwrap_or(OutputKind::Normal)
    }

    fn inplace_output(
        &mut self,
        context: &mut Context<'_, JitFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorDescription,
        input_index: usize,
    ) {
        let potential_inplace = plan.potential_inplaces.remove(input_index);
        let handle_input = plan.handle_inputs.get(potential_inplace.input_pos).unwrap();

        if plan.reference.is_none() {
            let index_input = self
                .inputs
                .get_index(output.precision, potential_inplace.tensor_relative.id)
                .unwrap();

            plan.reference = Some(Reference {
                layout: Arg::Input(index_input as u32, output.precision, LayoutInfo::IsRef),
                shape: tensor_global.shape.clone(),
                strides: handle_input.handle.strides.clone(),
            });

            if let Some(ops) = plan.reads.get_mut(&handle_input.relative_id) {
                for op in ops.iter_mut() {
                    if let ElemwiseOp::Assign(op) = op {
                        op.input.add_layout_info(LayoutInfo::IsRef);
                    };
                }
            }

            if let Some(ElemwiseOp::Assign(op)) = plan.writes.get_mut(&output.tensor_relative.id) {
                op.out.add_layout_info(LayoutInfo::IsRef);
            };
        }

        context
            .handles
            .register_handle(tensor_global.id, handle_input.handle.clone());

        self.handles[output.pos_original] = Some(HandleOutput::Alias {
            input_pos: potential_inplace.input_pos,
            precision: output.precision,
        });
        self.globals[output.pos_original] = Some(tensor_global);
    }

    #[allow(clippy::too_many_arguments)]
    fn normal_output<BT: BoolElement>(
        &mut self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, JitFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorDescription,
        strides: Vec<usize>,
    ) {
        if plan.reference.is_none() {
            let position = self
                .mapper
                .resolve_index(&output.precision, output.pos_original);
            plan.reference = Some(Reference {
                layout: Arg::Output(position, output.precision, LayoutInfo::IsRef),
                shape: tensor_global.shape.clone(),
                strides: strides.clone(),
            });

            if let ElemwiseOp::Assign(op) = plan.writes.get_mut(&output.tensor_relative.id).unwrap()
            {
                op.out.add_layout_info(LayoutInfo::IsRef);
            };
        } else if let Some(reference) = plan.reference.as_ref() {
            if reference.strides == strides && reference.shape == tensor_global.shape {
                if let ElemwiseOp::Assign(op) =
                    plan.writes.get_mut(&output.tensor_relative.id).unwrap()
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

        plan.rank = usize::max(tensor_global.shape.len(), plan.rank);
        context
            .handles
            .register_handle(tensor_global.id, handle.clone());

        self.handles[output.pos_original] = Some(HandleOutput::Owned {
            precision: output.precision,
            handle,
            global_shape: tensor_global.shape.clone(),
            global_id: tensor_global.id,
            vectorization: 1,
        });
        self.globals[output.pos_original] = Some(tensor_global);
    }

    #[allow(clippy::too_many_arguments)]
    fn reshaped_output<BT: BoolElement>(
        &mut self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, JitFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorDescription,
        strides: Vec<usize>,
        reshape: Reshape,
    ) {
        let original_handle = plan
            .handle_inputs
            .iter()
            .find(|handle| handle.relative_id == reshape.original)
            .unwrap();

        // We encode bool tensors as `B`.
        let dtype = match tensor_global.dtype {
            DType::Bool => BT::dtype(),
            _ => tensor_global.dtype,
        };

        if is_contiguous(
            &original_handle.global_shape,
            &original_handle.handle.strides,
        ) {
            plan.writes.remove(&output.tensor_relative.id);

            let handle = JitFusionHandle {
                client: client.clone(),
                handle: original_handle.handle.handle.clone(),
                device: device.clone(),
                strides,
                dtype,
            };
            context
                .handles
                .register_handle(tensor_global.id, handle.clone());
            // IT will never be access, just a way to keep the original position working.
            self.handles[output.pos_original] = Some(HandleOutput::Alias {
                input_pos: 0,
                precision: output.precision,
            });
            self.globals[output.pos_original] = Some(tensor_global);
        } else {
            self.normal_output::<BT>(
                client,
                device,
                context,
                plan,
                output,
                tensor_global,
                strides,
            );
        }
    }
}

/// Group output position by [element precision](ElemwisePrecision).
#[derive(Default, Debug)]
pub struct OutputPositionMapper {
    map: BTreeMap<ElemwisePrecision, Vec<usize>>,
}

impl OutputPositionMapper {
    /// Register a new output with the given precision and position.
    pub fn register(&mut self, precision: ElemwisePrecision, pos_handle: usize) {
        if let Some(positions) = self.map.get_mut(&precision) {
            positions.push(pos_handle);
        } else {
            self.map.insert(precision, vec![pos_handle]);
        }
    }

    /// Returns the right position from the precision and the global position in all outputs.
    pub fn resolve_index(&mut self, precision: &ElemwisePrecision, pos_handle: usize) -> u32 {
        self.map
            .get(precision)
            .unwrap()
            .iter()
            .enumerate()
            .find(|(_pos_elem, pos_all)| **pos_all == pos_handle)
            .map(|(pos_elem, _pos_all)| pos_elem)
            .unwrap() as u32
    }
}
