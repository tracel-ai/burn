use burn_common::tensor::{ReshapeAction, contiguous_strides, is_contiguous, reshape_action};
use burn_fusion::stream::Context;
use burn_ir::{TensorId, TensorIr};
use burn_tensor::DType;
use cubecl::{CubeElement, Runtime, client::ComputeClient, ir::Elem};

use crate::{
    CubeFusionHandle, elem_dtype,
    shared::{
        ir::{Arg, FuseOp, LayoutInfo},
        settings::RefLayoutSetting,
        trace::HandleInput,
    },
    strides_dyn_rank,
};

use super::{
    super::ir::FusePrecision, BlockPlan, FuseResources, HandleOutput, InputReference, LaunchPlan,
    NormalHandleInput, ReferenceSelection, RegisterTensor, TensorView, block::FuseBlock,
};

/// Create or reuse handles for the outputs.
///
/// It is also responsible to select the reference tensor.
pub struct OutputPlanner<'a, R: Runtime> {
    resources: &'a FuseResources,
    outputs_sorted: Vec<OutputSorted<'a>>,
    handles: Vec<Option<HandleOutput<R>>>,
    globals: Vec<Option<TensorIr>>,
    blocks: &'a Vec<FuseBlock>,
}

#[derive(Debug)]
struct OutputSorted<'a> {
    pos_original: usize,
    precision: FusePrecision,
    tensor_relative: &'a TensorIr,
}

#[derive(Debug)]
enum OutputKind {
    Normal,
    Inplace {
        /// The position in the potential inplace vector
        input_pos: usize,
    },
    Transform(TensorView),
}

impl<'a, R: Runtime> OutputPlanner<'a, R> {
    pub fn new(resources: &'a FuseResources, blocks: &'a Vec<FuseBlock>) -> Self {
        let mut outputs_sorted: Vec<_> = resources
            .outputs
            .iter()
            .enumerate()
            .filter_map(|(pos, entry)| match entry {
                RegisterTensor::Normal(ir, p) => Some((pos, ir, p)),
                RegisterTensor::QuantValues(_) => None,
                RegisterTensor::QuantParams(_) => None,
            })
            .map(|(pos, tensor, precision)| OutputSorted {
                pos_original: pos,
                precision: *precision,
                tensor_relative: tensor,
            })
            .collect();

        outputs_sorted.sort_by(|a, b| {
            let a_val: usize = a.tensor_relative.shape.iter().sum();
            let b_val: usize = b.tensor_relative.shape.iter().sum();

            b_val.cmp(&a_val)
        });

        let mut handles = Vec::with_capacity(resources.outputs.len());
        let mut globals = Vec::with_capacity(resources.outputs.len());

        for _ in 0..resources.outputs.len() {
            handles.push(None);
            globals.push(None);
        }

        Self {
            resources,
            outputs_sorted,
            handles,
            globals,
            blocks,
        }
    }

    pub fn run<BT: CubeElement>(
        mut self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, CubeFusionHandle<R>>,
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
            let (kind, block_idx) = self.output_kind(plan, &tensor_global, &output, &strides);

            match kind {
                OutputKind::Inplace { input_pos } => {
                    self.inplace_output(context, plan, output, tensor_global, input_pos, block_idx);
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
                        block_idx,
                    );
                }
                OutputKind::Transform(TensorView::Reshape { original, .. }) => {
                    self.reshaped_output::<BT>(
                        client,
                        device,
                        context,
                        plan,
                        output,
                        tensor_global,
                        strides,
                        original,
                        block_idx,
                    );
                }
                OutputKind::Transform(TensorView::SwapDims { original, dims, .. }) => {
                    self.swapped_dims_output::<BT>(
                        client,
                        device,
                        context,
                        plan,
                        output,
                        tensor_global,
                        original,
                        dims,
                        block_idx,
                    );
                }
            }
        }

        for (handle, global) in self.handles.into_iter().zip(self.globals.into_iter()) {
            plan.handle_outputs.push(handle.unwrap());
            plan.global_outputs.push(global.unwrap());
        }

        for (i, block) in plan.blocks.iter_mut().enumerate() {
            if !block.reference.is_found() {
                Self::select_reference_from_inputs(
                    self.blocks[i].settings.ref_layout,
                    block,
                    &plan.handle_inputs,
                );
            } else {
                Self::add_layout_info_inputs(block, &plan.handle_inputs);
            }
        }

        // Make sure dropped are correctly executed.
        for id in self.resources.dropped.iter() {
            if let Some(tensor_global) = context.tensors.get(id) {
                context.handles.remove_handle(tensor_global.id);
            }
        }
        for id in plan.cleared.drain(..) {
            context.handles.remove_handle(id);
        }
    }

    fn select_reference_from_inputs(
        ref_layout_setting: RefLayoutSetting,
        block: &mut BlockPlan<'_>,
        handle_inputs: &[HandleInput<R>],
    ) {
        if let Some(input_ref) = block.potential_reference_input.take() {
            match input_ref {
                InputReference::Normal { input_pos } => {
                    let reference = handle_inputs
                        .get(input_pos)
                        .unwrap()
                        .as_normal()
                        .expect("Quant can't be used as inplace");

                    let set_ref_as_concrete = |block: &mut BlockPlan<'_>| {
                        block.reference = ReferenceSelection::Concrete {
                            layout: Arg::Input(
                                input_pos as u32,
                                reference.precision,
                                LayoutInfo::IsRef,
                            ),
                            shape: reference.global_ir.shape.clone(),
                            strides: reference.handle.strides.clone(),
                        };
                    };

                    let set_ref_as_virtual = |block: &mut BlockPlan<'_>| {
                        block.reference = ReferenceSelection::VirtualShape {
                            original: Arg::Input(
                                input_pos as u32,
                                reference.precision,
                                LayoutInfo::Unknown,
                            ),
                            shape: reference.global_ir.shape.clone(),
                            strides: contiguous_strides(&reference.global_ir.shape),
                        };
                    };

                    match ref_layout_setting {
                        RefLayoutSetting::Any => set_ref_as_concrete(block),
                        RefLayoutSetting::OnlyContiguous => {
                            if is_contiguous(&reference.global_ir.shape, &reference.handle.strides)
                            {
                                set_ref_as_concrete(block)
                            } else {
                                set_ref_as_virtual(block)
                            }
                        }
                    }

                    Self::add_layout_info_inputs(block, handle_inputs);
                }
                InputReference::SwapDims { original_pos, dims } => {
                    let reference = handle_inputs
                        .get(original_pos)
                        .unwrap()
                        .as_normal()
                        .expect("Quant can't be used in swap dims operation");
                    block.reference = ReferenceSelection::SwapDims {
                        original: Arg::Input(
                            original_pos as u32,
                            reference.precision,
                            LayoutInfo::Unknown,
                        ),
                        dims,
                    };
                }
                InputReference::Reshaped { reshape_pos } => {
                    block.reference = ReferenceSelection::Reshaped { reshape_pos };
                }
            };
        } else {
            block.reference = ReferenceSelection::NotFound;
        }
    }

    fn add_layout_info_inputs(block: &mut BlockPlan<'_>, handle_inputs: &[HandleInput<R>]) {
        for hi in handle_inputs.iter().filter_map(|h| match h {
            HandleInput::Normal(input) => Some(input),
            _ => None,
        }) {
            let (strides, shape) = match &block.reference {
                ReferenceSelection::Concrete { strides, shape, .. }
                | ReferenceSelection::VirtualShape { strides, shape, .. } => (strides, shape),
                _ => continue,
            };

            if strides == &hi.handle.strides
                && shape == &hi.global_ir.shape
                && let Some(ops) = block.reads.get_mut(&hi.relative_id)
            {
                for op in ops.iter_mut() {
                    if let FuseOp::Assign(op) = op {
                        op.input.add_layout_info(LayoutInfo::SameAsRef);
                    }
                }
            }
        }
    }

    fn output_kind(
        &self,
        plan: &mut LaunchPlan<'a, R>,
        tensor_global: &TensorIr,
        output: &OutputSorted,
        strides: &[usize],
    ) -> (OutputKind, usize) {
        let mut block_idx = None;
        for (i, block) in plan.blocks.iter().enumerate() {
            if block.writes.contains_key(&output.tensor_relative.id) {
                block_idx = Some(i);
                break;
            }
        }
        let block_idx = block_idx.unwrap();

        if let Some(transform) = self.resources.views.iter().find(|v| match v {
            TensorView::Reshape { reshaped, .. } => reshaped == &output.tensor_relative.id,
            TensorView::SwapDims { swapped, .. } => swapped == &output.tensor_relative.id,
        }) {
            return (OutputKind::Transform(transform.clone()), block_idx);
        }

        let block = &plan.blocks[block_idx];
        let kind = block
            .potential_inplaces
            .iter()
            .enumerate()
            .find(|(_pos, pi)| {
                pi.tensor_relative.dtype == tensor_global.dtype
                    && pi.tensor_relative.shape == output.tensor_relative.shape
                    && pi.strides == strides
                    && block.reference.compatible_strides_for_inplace(strides)
            })
            .map(|(pos, _)| OutputKind::Inplace { input_pos: pos })
            .unwrap_or(OutputKind::Normal);

        (kind, block_idx)
    }

    fn inplace_output(
        &mut self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorIr,
        input_index: usize,
        block_idx: usize,
    ) {
        let block = &mut plan.blocks[block_idx];
        let potential_inplace = block.potential_inplaces.remove(input_index);
        let handle_input = match plan.handle_inputs.get(potential_inplace.input_pos).unwrap() {
            HandleInput::Normal(handle) => handle,
            _ => {
                unreachable!("Quant tensor handle can't be used inplace yet.")
            }
        };

        if !block.reference.is_found() {
            let index_input = self
                .resources
                .inputs
                .get_index(potential_inplace.tensor_relative.id)
                .unwrap();

            block.reference = ReferenceSelection::Concrete {
                layout: Arg::Input(index_input, output.precision, LayoutInfo::IsRef),
                shape: tensor_global.shape.clone(),
                strides: handle_input.handle.strides.clone(),
            };

            if let Some(ops) = block.reads.get_mut(&handle_input.relative_id) {
                for op in ops.iter_mut() {
                    if let FuseOp::Assign(op) = op {
                        op.input.add_layout_info(LayoutInfo::IsRef);
                    };
                }
            }

            if let Some(FuseOp::Assign(op)) = block.writes.get_mut(&output.tensor_relative.id) {
                op.out.add_layout_info(LayoutInfo::IsRef);
            };
        } else {
            // Already validated, necessary for correctness.
            if let Some(FuseOp::Assign(op)) = block.writes.get_mut(&output.tensor_relative.id) {
                op.out.add_layout_info(LayoutInfo::SameAsRef);
            };
        }

        context
            .handles
            .register_handle(tensor_global.id, handle_input.handle.clone());

        self.handles[output.pos_original] = Some(HandleOutput::Alias {
            input_pos: potential_inplace.input_pos,
            precision: output.precision,
            #[cfg(feature = "autotune-checks")]
            debug_info: super::HandleOutputAliasDebugInfo {
                relative_id: output.tensor_relative.id,
                handle: handle_input.handle.clone(),
                global_shape: tensor_global.shape.clone(),
            },
        });
        self.globals[output.pos_original] = Some(tensor_global);
    }

    #[allow(clippy::too_many_arguments)]
    fn normal_output<BT: CubeElement>(
        &mut self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorIr,
        strides: Vec<usize>,
        block_idx: usize,
    ) {
        let block = &mut plan.blocks[block_idx];

        if !block.reference.is_found()
            && self.blocks[block_idx].shape_ref == output.tensor_relative.shape
        {
            block.reference = ReferenceSelection::Concrete {
                layout: Arg::Output(
                    output.pos_original as u32,
                    output.precision,
                    LayoutInfo::IsRef,
                ),
                shape: tensor_global.shape.clone(),
                strides: strides.clone(),
            };

            // Sometimes outputs that are manually handled don't have any write registered.
            if let Some(FuseOp::Assign(op)) = block.writes.get_mut(&output.tensor_relative.id) {
                op.out.add_layout_info(LayoutInfo::IsRef);
            };
        } else if let ReferenceSelection::Concrete {
            shape: ref_shape,
            strides: ref_strides,
            ..
        } = &block.reference
            && ref_strides == &strides
            && ref_shape == &tensor_global.shape
            && let FuseOp::Assign(op) = block.writes.get_mut(&output.tensor_relative.id).unwrap()
        {
            op.out.add_layout_info(LayoutInfo::SameAsRef);
        };

        // We encode bool tensors as `B`.
        let dtype = match tensor_global.dtype {
            DType::Bool => elem_dtype::<BT>(),
            _ => tensor_global.dtype,
        };
        let size = tensor_global.shape.iter().product::<usize>() * Elem::from(dtype).size();

        let handle = CubeFusionHandle {
            client: client.clone(),
            handle: client.empty(size),
            device: device.clone(),
            strides,
            dtype,
            qparams: None,
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
            relative_id: output.tensor_relative.id,
            vectorization: 1,
        });
        self.globals[output.pos_original] = Some(tensor_global);
    }

    #[allow(clippy::too_many_arguments)]
    fn reshaped_output<BT: CubeElement>(
        &mut self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorIr,
        strides: Vec<usize>,
        original: TensorId,
        block_idx: usize,
    ) {
        let block = &mut plan.blocks[block_idx];

        let (pos_input, original_handle) = Self::find_child_input(&plan.handle_inputs, original);

        // We encode bool tensors as `B`.
        let dtype = match tensor_global.dtype {
            DType::Bool => elem_dtype::<BT>(),
            _ => tensor_global.dtype,
        };

        let action = reshape_action(
            &original_handle.global_ir.shape,
            &original_handle.handle.strides,
            &tensor_global.shape,
        );

        let update = match action {
            ReshapeAction::UpdateStrides { strides } => Some(strides),
            ReshapeAction::NoChange => Some(original_handle.handle.strides.clone()),
            ReshapeAction::Recompute => None,
        };

        match update {
            Some(strides) => {
                block.writes.remove(&output.tensor_relative.id);

                let handle = CubeFusionHandle {
                    client: client.clone(),
                    handle: original_handle.handle.handle.clone(),
                    device: device.clone(),
                    strides,
                    dtype,
                    qparams: original_handle.handle.qparams.clone(),
                };
                context
                    .handles
                    .register_handle(tensor_global.id, handle.clone());

                // IT will never be access, just a way to keep the original position working.
                self.handles[output.pos_original] = Some(HandleOutput::Alias {
                    input_pos: pos_input,
                    precision: output.precision,
                    #[cfg(feature = "autotune-checks")]
                    debug_info: super::HandleOutputAliasDebugInfo {
                        relative_id: output.tensor_relative.id,
                        handle: handle.clone(),
                        global_shape: tensor_global.shape.clone(),
                    },
                });
                self.globals[output.pos_original] = Some(tensor_global);
            }
            None => {
                self.normal_output::<BT>(
                    client,
                    device,
                    context,
                    plan,
                    output,
                    tensor_global,
                    strides,
                    block_idx,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn swapped_dims_output<BT: CubeElement>(
        &mut self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorIr,
        original: TensorId,
        dims: (u32, u32),
        block_idx: usize,
    ) {
        let block = &mut plan.blocks[block_idx];
        let (pos_input, original_handle) = Self::find_child_input(&plan.handle_inputs, original);

        // We encode bool tensors as `B`.
        let dtype = match tensor_global.dtype {
            DType::Bool => elem_dtype::<BT>(),
            _ => tensor_global.dtype,
        };

        // TODO: Check if we can also remove the read, if we have a dead partial graph.
        block.writes.remove(&output.tensor_relative.id);

        let strides = original_handle.handle.strides.clone();

        let mut handle = CubeFusionHandle {
            client: client.clone(),
            handle: original_handle.handle.handle.clone(),
            device: device.clone(),
            strides,
            dtype,
            qparams: original_handle.handle.qparams.clone(),
        };
        handle.strides.swap(dims.0 as usize, dims.1 as usize);

        context
            .handles
            .register_handle(tensor_global.id, handle.clone());

        // IT will never be access, just a way to keep the original position working.
        self.handles[output.pos_original] = Some(HandleOutput::Alias {
            input_pos: pos_input,
            precision: output.precision,
            #[cfg(feature = "autotune-checks")]
            debug_info: super::HandleOutputAliasDebugInfo {
                relative_id: output.tensor_relative.id,
                handle: handle.clone(),
                global_shape: tensor_global.shape.clone(),
            },
        });
        self.globals[output.pos_original] = Some(tensor_global);
    }

    fn find_child_input(
        handle_inputs: &[HandleInput<R>],
        original: TensorId,
    ) -> (usize, &NormalHandleInput<R>) {
        handle_inputs
            .iter()
            .enumerate()
            .find_map(|(pi, handle)| match handle {
                HandleInput::Normal(handle) => match handle.relative_id == original {
                    true => Some((pi, handle)),
                    false => None,
                },
                _ => None, // Quant tensor can't be reshaped.
            })
            .unwrap()
    }
}
