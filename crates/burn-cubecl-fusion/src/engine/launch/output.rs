use super::{
    super::codegen::ir::FuseType, BlockPlan, HandleOutput, InputReference, LaunchPlan,
    NormalHandleInput, ReferenceSelection,
};
use crate::{
    CubeFusionHandle,
    engine::{
        codegen::ir::{FuseArg, FuseOp, LayoutInfo},
        launch::{
            HandleInput,
            layout::{DimOrder, dim_order, is_contiguous_order, strides_for},
        },
        settings::{FuseSettings, RefLayoutSetting},
        trace::{FuseResources, RegisterTensor, RuntimeLayout, TensorView, block::FuseBlock},
    },
    strides_dyn_rank,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_fusion::stream::Context;
use burn_ir::{TensorId, TensorIr};
use burn_std::Shape;
use burn_std::{
    Strides,
    tensor::{ReshapeAction, contiguous_strides, is_contiguous, is_dense, reshape_action},
};
use cubecl::{Runtime, client::ComputeClient};

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
    precision: FuseType,
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

    pub fn run(
        mut self,
        client: &ComputeClient<R>,
        device: &R::Device,
        context: &mut Context<CubeFusionHandle<R>>,
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
            let (kind, block_idx) = self.output_kind(plan, &tensor_global, &output);

            match kind {
                OutputKind::Inplace { input_pos } => {
                    self.inplace_output(context, plan, output, tensor_global, input_pos, block_idx);
                }
                OutputKind::Normal => {
                    // A normal output has no layout forced on it by a view, so the
                    // block is free to lay it out in whatever order costs its inputs
                    // the least traffic. The transform kinds below keep the strides
                    // their view dictates.
                    let strides = self.chosen_strides(
                        plan,
                        block_idx,
                        &tensor_global.shape,
                        &output.tensor_relative.shape,
                    );

                    self.normal_output(
                        client,
                        device,
                        context,
                        plan,
                        output,
                        tensor_global,
                        strides,
                        LayoutInfo::IsRef,
                        block_idx,
                    );
                }
                OutputKind::Transform(TensorView::Reshape { original, .. }) => {
                    self.reshaped_output(
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
                    self.swapped_dims_output(
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
                OutputKind::Transform(TensorView::NhwcStrides {
                    stride_relayout, ..
                }) => {
                    self.nhwc_strides_output(
                        client,
                        device,
                        context,
                        plan,
                        output,
                        tensor_global,
                        strides,
                        block_idx,
                        stride_relayout,
                    );
                }
            }
        }

        for (handle, global) in self.handles.into_iter().zip(self.globals) {
            plan.handle_outputs.push(handle.unwrap());
            plan.global_outputs.push(global.unwrap());
        }

        for i in 0..plan.blocks.len() {
            if !plan.blocks[i].reference.is_found() {
                match self.blocks[i].settings.ref_layout {
                    RefLayoutSetting::SameAsBlock { block_pos } => {
                        plan.blocks[i].reference =
                            plan.blocks[block_pos as usize].reference.clone();
                    }
                    _ => {
                        let new_runtime = Self::select_reference_from_inputs(
                            &self.blocks[i],
                            &mut plan.blocks[i],
                            &plan.handle_inputs,
                        );

                        if let Some(shape) = new_runtime {
                            let pos = plan.runtime_layouts.len();
                            let mut shape_global = shape.clone();
                            for (i, s) in shape.iter().enumerate() {
                                shape_global[i] = *context.shapes_relative2global.get(s).expect(
                                    "reference shape ids to be assigned by the running stream",
                                );
                            }

                            let strides = strides_dyn_rank(&shape_global);

                            plan.blocks[i].reference = ReferenceSelection::Runtime { pos };
                            plan.runtime_layouts.push(RuntimeLayout {
                                shape: shape_global,
                                strides,
                            });
                        }
                    }
                };
            } else {
                Self::add_layout_info_inputs(&mut plan.blocks[i], &plan.handle_inputs);
            }
        }

        // Make sure dropped are correctly executed.
        for id in self.resources.dropped.iter() {
            if let Some(tensor_global) = context.tensors.get(id) {
                context.handles.remove_handle(tensor_global.id);
            }
        }
    }

    fn select_reference_from_inputs(
        block: &FuseBlock,
        block_plan: &mut BlockPlan<'_>,
        handle_inputs: &[HandleInput<R>],
    ) -> Option<Shape> {
        if let Some(input_ref) = block_plan.potential_reference_input.take() {
            match input_ref {
                InputReference::Normal { input_pos } => {
                    let reference = handle_inputs
                        .get(input_pos)
                        .unwrap()
                        .as_normal()
                        .expect("Quant can't be used as inplace");

                    let set_ref_as_concrete = |block: &mut BlockPlan<'_>| {
                        block.reference = ReferenceSelection::Concrete {
                            layout: FuseArg::Input(
                                input_pos,
                                reference.precision,
                                LayoutInfo::IsRef,
                            ),
                            shape: reference.global_ir.shape.clone(),
                            strides: reference.handle.strides.clone(),
                        };
                    };

                    let set_ref_as_virtual = |block: &mut BlockPlan<'_>| {
                        block.reference = ReferenceSelection::VirtualShape {
                            original: FuseArg::Input(
                                input_pos,
                                reference.precision,
                                LayoutInfo::Unknown,
                            ),
                            shape: reference.global_ir.shape.clone(),
                            strides: contiguous_strides(&reference.global_ir.shape),
                        };
                    };

                    match block.settings.ref_layout {
                        RefLayoutSetting::Any => {
                            // A padded reference would make the kernel walk a prefix of the
                            // buffer: the launch is sized from the logical element count while a
                            // position indexes the buffer.
                            if is_dense(&reference.global_ir.shape, &reference.handle.strides) {
                                set_ref_as_concrete(block_plan)
                            } else {
                                set_ref_as_virtual(block_plan)
                            }
                        }
                        RefLayoutSetting::SameAsBlock { .. } => {
                            // Skip set ref.
                        }
                        RefLayoutSetting::OnlyContiguous => {
                            if is_contiguous(&reference.global_ir.shape, &reference.handle.strides)
                            {
                                set_ref_as_concrete(block_plan)
                            } else {
                                set_ref_as_virtual(block_plan)
                            }
                        }
                    }

                    Self::add_layout_info_inputs(block_plan, handle_inputs);
                }
                InputReference::SwapDims { original_pos, dims } => {
                    let reference = handle_inputs
                        .get(original_pos)
                        .unwrap()
                        .as_normal()
                        .expect("Quant can't be used in swap dims operation");
                    block_plan.reference = ReferenceSelection::SwapDims {
                        original: FuseArg::Input(
                            original_pos,
                            reference.precision,
                            LayoutInfo::Unknown,
                        ),
                        dims,
                    };
                }
                InputReference::Reshaped { reshape_pos } => {
                    block_plan.reference = ReferenceSelection::Reshaped { reshape_pos };
                }
            };
            None
        } else {
            Some(block.shape_ref.clone())
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

    /// The strides to allocate a normal output of this block with.
    ///
    /// The reference layout is taken from the first output allocated, so this is
    /// what decides how the whole block iterates. Contiguous strides in logical
    /// dimension order — what this used to be, unconditionally — are the right
    /// answer only when the inputs are contiguous too, and after a convolution
    /// they are not.
    fn chosen_strides(
        &self,
        plan: &LaunchPlan<'a, R>,
        block_idx: usize,
        shape: &Shape,
        shape_relative: &Shape,
    ) -> Strides {
        // Once a reference exists, the remaining outputs follow it, so their writes
        // are linear as well. Their shapes can differ from the reference's; the
        // dimension *order* is what carries over.
        if let ReferenceSelection::Concrete {
            shape: ref_shape,
            strides: ref_strides,
            ..
        } = &plan.blocks[block_idx].reference
            && ref_shape.num_dims() == shape.num_dims()
            && let Some(order) = dim_order(ref_shape, ref_strides)
        {
            return strides_for(shape, &order);
        }

        // Only the output that goes on to *be* the reference gets to pick a layout.
        // Any other output is written against a reference it does not define, so a
        // layout of its own would leave its own writes strided and nothing else
        // improved — and would hand the vectorization planner an output whose
        // innermost dimension is not the one the block iterates along.
        // [Self::normal_output] selects the reference on exactly this condition.
        if &self.blocks[block_idx].shape_ref != shape_relative {
            return strides_dyn_rank(shape);
        }

        match self.preferred_dim_order(plan, block_idx, shape) {
            Some(order) => strides_for(shape, &order),
            None => strides_dyn_rank(shape),
        }
    }

    /// The dimension order that costs this block's inputs the least strided
    /// traffic, or `None` to keep the contiguous one.
    ///
    /// Writing an output in any dense order costs the same, so the output layout
    /// is free and the only price of a choice is the inputs that disagree with
    /// it. That makes the best choice the plurality of the inputs, weighted by
    /// the bytes each one moves.
    ///
    /// Only inputs of exactly this shape vote. A broadcast parameter is read from
    /// cache whatever the order is, and a differently shaped input can never be
    /// `SameAsRef` regardless, so neither has a stake in the outcome.
    ///
    /// Because a convolution's output is NHWC and its consumers' outputs then
    /// become NHWC too, this propagates forward on its own: a `conv -> norm ->
    /// silu -> conv` chain settles into NHWC end to end, and the permutes around
    /// each convolution stay the metadata changes they are meant to be.
    fn preferred_dim_order(
        &self,
        plan: &LaunchPlan<'a, R>,
        block_idx: usize,
        shape: &Shape,
    ) -> Option<DimOrder> {
        if !may_permute_layout(
            &self.blocks[block_idx].settings,
            &self.blocks[block_idx].ops,
        ) {
            return None;
        }

        let block = &plan.blocks[block_idx];
        let mut votes: Vec<(DimOrder, usize)> = Vec::new();

        for input in plan.handle_inputs.iter() {
            let Some(input) = input.as_normal() else {
                // A quantized input cannot be `SameAsRef`.
                continue;
            };

            if !block.reads.contains_key(&input.relative_id)
                || self.resources.indexed.contains_key(&input.relative_id)
                || self.resources.inputs_unhandled.contains(&input.relative_id)
                || &input.global_ir.shape != shape
            {
                continue;
            }

            let Some(order) = dim_order(shape, &input.handle.strides) else {
                // Not dense: sliced, broadcast, or otherwise not describable as an
                // order. The block cannot adopt a layout it cannot express.
                continue;
            };

            let bytes = shape.num_elements() * dtype_to_storage_type(input.global_ir.dtype).size();

            match votes.iter_mut().find(|(candidate, _)| candidate == &order) {
                Some((_, total)) => *total += bytes,
                None => votes.push((order, bytes)),
            }
        }

        let winner = votes.into_iter().max_by_key(|(order, bytes)| {
            // Ties go to the contiguous order, so a block whose inputs disagree
            // keeps behaving the way it always has.
            (*bytes, is_contiguous_order(order))
        })?;

        match is_contiguous_order(&winner.0) {
            true => None,
            false => Some(winner.0),
        }
    }

    fn output_kind(
        &self,
        plan: &mut LaunchPlan<'a, R>,
        tensor_global: &TensorIr,
        output: &OutputSorted,
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
            TensorView::NhwcStrides { id, .. } => id == &output.tensor_relative.id,
        }) {
            return (OutputKind::Transform(transform.clone()), block_idx);
        }

        let block = &plan.blocks[block_idx];
        let ref_layout_setting = &self.blocks[block_idx].settings.ref_layout;
        let may_permute = may_permute_layout(
            &self.blocks[block_idx].settings,
            &self.blocks[block_idx].ops,
        );
        let kind = block
            .potential_inplaces
            .iter()
            .enumerate()
            .find(|(_pos, pi)| {
                pi.tensor_relative.dtype == tensor_global.dtype
                    && pi.tensor_relative.shape == output.tensor_relative.shape
                    // The candidate only has to be *dense*, not contiguous. Requiring
                    // contiguity here made every convolution output ineligible — a
                    // convolution hands over an NCHW view of NHWC memory — so the one
                    // buffer whose layout the block should have adopted was the one
                    // buffer it always refused, and the block allocated a contiguous
                    // output and read its input strided instead.
                    && dim_order(&tensor_global.shape, &pi.strides).is_some()
                    && if block.reference.is_found() {
                        // An already-selected reference must have compatible strides.
                        // Compared against the candidate's own strides, since those are
                        // what the aliased buffer actually has.
                        block.reference.compatible_strides_for_inplace(&pi.strides)
                    } else if may_permute {
                        // When no reference has been selected yet, this output becomes
                        // the reference (see [Self::inplace_output]); requiring an
                        // existing reference here made the first output of every block
                        // ineligible, since the reference is only selected while
                        // processing outputs. This block can iterate in the candidate's
                        // order, so any dense layout will do.
                        true
                    } else {
                        // Aliasing here makes the candidate's layout the block's
                        // reference, and this block is not one that can iterate in a
                        // permuted order — so let it in only when it is contiguous,
                        // which is what this check was before dense candidates were
                        // allowed at all. A block that inherits its reference from
                        // another cannot validate one here regardless.
                        !matches!(ref_layout_setting, RefLayoutSetting::SameAsBlock { .. })
                            && is_contiguous(&tensor_global.shape, &pi.strides)
                    }
            })
            .map(|(pos, _)| OutputKind::Inplace { input_pos: pos })
            .unwrap_or(OutputKind::Normal);

        (kind, block_idx)
    }

    #[allow(clippy::too_many_arguments)]
    fn inplace_output(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorIr,
        input_index: usize,
        block_idx: usize,
    ) {
        let block = &mut plan.blocks[block_idx];
        #[cfg(feature = "test-util")]
        crate::inspect::record_inplace_alias();
        let potential_inplace = block.potential_inplaces.remove(input_index);
        let handle_input = match plan.handle_inputs.get(potential_inplace.input_pos).unwrap() {
            HandleInput::Normal(handle) => handle,
            _ => {
                unreachable!("Quant tensor handle can't be used inplace yet.")
            }
        };

        // [Self::output_kind] only selects inplace when the reference is already
        // validated or this output can become the reference, which blocks inheriting
        // their reference from another block (`SameAsBlock`) never can.
        debug_assert!(
            block.reference.is_found()
                || !matches!(
                    self.blocks[block_idx].settings.ref_layout,
                    RefLayoutSetting::SameAsBlock { .. }
                ),
            "inplace alias selected for a `SameAsBlock` block without a validated reference",
        );

        if !block.reference.is_found() {
            // The aliased output shares the input's buffer and layout, so the reference
            // is expressed with the output argument. Runners assume references are
            // either output-concrete or virtual; an input-concrete reference here would
            // be resolved against the wrong argument list.
            block.reference = ReferenceSelection::Concrete {
                layout: FuseArg::Output(output.pos_original, output.precision, LayoutInfo::IsRef),
                shape: tensor_global.shape.clone(),
                strides: handle_input.handle.strides.clone(),
            };

            if let Some(ops) = block.reads.get_mut(&handle_input.relative_id) {
                for op in ops.iter_mut() {
                    if let FuseOp::Assign(op) = op {
                        op.input.add_layout_info(LayoutInfo::IsRef);
                        break;
                    };
                }
            }

            if let Some(ops) = block.writes.get_mut(&output.tensor_relative.id) {
                for op in ops {
                    if let FuseOp::Assign(op) = op {
                        op.out.add_layout_info(LayoutInfo::IsRef);
                        break;
                    }
                }
            };
        } else {
            // Already validated, necessary for correctness.
            if let Some(ops) = block.writes.get_mut(&output.tensor_relative.id) {
                for op in ops {
                    if let FuseOp::Assign(op) = op {
                        op.out.add_layout_info(LayoutInfo::SameAsRef);
                        break;
                    }
                }
            };
        }

        context
            .handles
            .register_handle(tensor_global.id, handle_input.handle.clone());

        self.handles[output.pos_original] = Some(HandleOutput::Alias {
            input_pos: potential_inplace.input_pos,
            precision: output.precision,
            global_shape: tensor_global.shape.clone(),
            // The aliased buffer keeps the input's layout, which is not contiguous
            // when the input came out of a convolution.
            strides: handle_input.handle.strides.clone(),
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
    fn normal_output(
        &mut self,
        client: &ComputeClient<R>,
        device: &R::Device,
        context: &mut Context<CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorIr,
        strides: Strides,
        base_layout_info: LayoutInfo,
        block_idx: usize,
    ) {
        let block = &mut plan.blocks[block_idx];

        let is_allowed_by_settings = match self.blocks[block_idx].settings.ref_layout {
            RefLayoutSetting::SameAsBlock { .. } => false,
            RefLayoutSetting::OnlyContiguous => is_contiguous(&tensor_global.shape, &strides),
            _ => true,
        };

        if !block.reference.is_found()
            && self.blocks[block_idx].shape_ref == output.tensor_relative.shape
            && base_layout_info != LayoutInfo::Unknown
            && is_allowed_by_settings
        {
            block.reference = ReferenceSelection::Concrete {
                layout: FuseArg::Output(output.pos_original, output.precision, base_layout_info),
                shape: tensor_global.shape.clone(),
                strides: strides.clone(),
            };

            // Sometimes outputs that are manually handled don't have any write registered.
            if let Some(ops) = block.writes.get_mut(&output.tensor_relative.id) {
                for op in ops {
                    if let FuseOp::Assign(op) = op {
                        op.out.add_layout_info(base_layout_info);
                        break;
                    }
                }
            };
        } else if let ReferenceSelection::Concrete {
            shape: ref_shape,
            strides: ref_strides,
            ..
        } = &block.reference
            && ref_strides == &strides
            && ref_shape == &tensor_global.shape
            && let Some(ops) = block.writes.get_mut(&output.tensor_relative.id)
        {
            for op in ops {
                if let FuseOp::Assign(op) = op {
                    op.out.add_layout_info(LayoutInfo::SameAsRef);
                    break;
                }
            }
        };

        let dtype = tensor_global.dtype;
        let size =
            tensor_global.shape.iter().product::<usize>() * dtype_to_storage_type(dtype).size();

        let handle = CubeFusionHandle {
            client: client.clone(),
            handle: client.empty(size),
            device: device.clone(),
            strides,
            dtype,
            qparams: None,
        };

        plan.rank = usize::max(tensor_global.shape.rank(), plan.rank);
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
    fn reshaped_output(
        &mut self,
        client: &ComputeClient<R>,
        device: &R::Device,
        context: &mut Context<CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorIr,
        strides: Strides,
        original: TensorId,
        block_idx: usize,
    ) {
        let block = &mut plan.blocks[block_idx];

        let (pos_input, original_handle) = Self::find_child_input(&plan.handle_inputs, original);

        let dtype = tensor_global.dtype;

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
                // We modify the metadata instead.
                remove_concrete_write(block, output.tensor_relative.id, output.pos_original);

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
                    global_shape: tensor_global.shape.clone(),
                    strides: handle.strides.clone(),
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
                self.normal_output(
                    client,
                    device,
                    context,
                    plan,
                    output,
                    tensor_global,
                    strides,
                    LayoutInfo::IsRef,
                    block_idx,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn swapped_dims_output(
        &mut self,
        client: &ComputeClient<R>,
        device: &R::Device,
        context: &mut Context<CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorIr,
        original: TensorId,
        dims: (usize, usize),
        block_idx: usize,
    ) {
        let block = &mut plan.blocks[block_idx];
        let (pos_input, original_handle) = Self::find_child_input(&plan.handle_inputs, original);

        let dtype = tensor_global.dtype;

        // TODO: Check if we can also remove the read, if we have a dead partial graph.
        //
        // We modify the metadata instead.
        remove_concrete_write(block, output.tensor_relative.id, output.pos_original);

        let strides = original_handle.handle.strides.clone();

        let mut handle = CubeFusionHandle {
            client: client.clone(),
            handle: original_handle.handle.handle.clone(),
            device: device.clone(),
            strides,
            dtype,
            qparams: original_handle.handle.qparams.clone(),
        };
        handle.strides.swap(dims.0, dims.1);

        context
            .handles
            .register_handle(tensor_global.id, handle.clone());

        // IT will never be access, just a way to keep the original position working.
        self.handles[output.pos_original] = Some(HandleOutput::Alias {
            input_pos: pos_input,
            precision: output.precision,
            global_shape: tensor_global.shape.clone(),
            strides: handle.strides.clone(),
            #[cfg(feature = "autotune-checks")]
            debug_info: super::HandleOutputAliasDebugInfo {
                relative_id: output.tensor_relative.id,
                handle: handle.clone(),
                global_shape: tensor_global.shape.clone(),
            },
        });
        self.globals[output.pos_original] = Some(tensor_global);
    }

    #[allow(clippy::too_many_arguments)]
    fn nhwc_strides_output(
        &mut self,
        client: &ComputeClient<R>,
        device: &R::Device,
        context: &mut Context<CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorIr,
        mut strides: Strides,
        block_idx: usize,
        stride_relayout: Shape,
    ) {
        let strides_changed =
            relayout_strides(&mut strides, &tensor_global.shape, &stride_relayout);

        let base_layout_info = if strides_changed {
            LayoutInfo::Unknown
        } else {
            LayoutInfo::IsRef
        };

        self.normal_output(
            client,
            device,
            context,
            plan,
            output,
            tensor_global,
            strides,
            base_layout_info,
            block_idx,
        );
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

/// Whether a block may iterate — and so write its outputs — in a permuted
/// dimension order.
///
/// Three things have to hold, and each one has bitten:
///
/// The settings must allow a free reference at all. A reduce pins it contiguous,
/// and a block that inherits its reference from another is not the one making the
/// decision.
///
/// The runner must read and write every operand through the generic fused paths.
/// The matmul one does not: it describes its output to the matmul algorithm as
/// row-major while building the output view from the reference's last two strides,
/// so a permuted reference has it writing lines that are not contiguous along the
/// column. That is what [FuseSettings::choose_output_layout] gates.
///
/// And the block must contain no operation that indexes its operands against the
/// last dimension directly rather than through the reference. `gather` walks its
/// lanes with the stride of `rank - 1`; it agrees with the rest of the kernel only
/// while the reference is contiguous.
fn may_permute_layout(settings: &FuseSettings, ops: &[FuseOp]) -> bool {
    if !matches!(settings.ref_layout, RefLayoutSetting::Any) || !settings.choose_output_layout {
        return false;
    }

    !ops.iter().any(|op| {
        matches!(
            op,
            FuseOp::Gather { .. }
                | FuseOp::Select { .. }
                | FuseOp::Cat { .. }
                | FuseOp::Dequantize { .. }
        )
    })
}

fn remove_concrete_write(block: &mut BlockPlan, id: TensorId, output_pos: usize) {
    let ops = block.writes.remove(&id);

    if let Some(ops) = ops {
        let mut keep = Vec::with_capacity(ops.len());

        for op in ops {
            if let FuseOp::Assign(args) = &op {
                if let FuseArg::Output(pos, ..) = args.out {
                    if pos != output_pos {
                        keep.push(op);
                    }
                } else {
                    keep.push(op);
                }
            }
        }
        block.writes.insert(id, keep);
    }
}

fn relayout_strides(strides: &mut Strides, shape: &Shape, stride_relayout: &Shape) -> bool {
    let rank = shape.num_dims();

    if rank < 2 || stride_relayout.num_dims() != rank {
        return false;
    }

    let mut dims_by_target_pos = vec![None; rank];

    for original_dim in 0..rank {
        let target_pos = stride_relayout[original_dim];

        if target_pos >= rank || dims_by_target_pos[target_pos].is_some() {
            return false;
        }

        dims_by_target_pos[target_pos] = Some(original_dim);
    }

    let mut current_stride = 1;
    let mut strides_changed = false;

    for original_dim in dims_by_target_pos.into_iter().rev().flatten() {
        if strides[original_dim] != current_stride {
            strides[original_dim] = current_stride;
            strides_changed = true;
        }
        current_stride *= shape[original_dim];
    }

    strides_changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{
        codegen::ir::{FuseType, QuantSchemeFuse, UnaryFuseArgs},
        settings::VectorizationSetting,
    };
    use cubecl::quant::scheme::QuantScheme;

    fn arg(pos: usize) -> FuseArg {
        FuseArg::Input(pos, FuseType::F32, LayoutInfo::Unknown)
    }

    fn elemwise_op() -> FuseOp {
        FuseOp::Assign(UnaryFuseArgs {
            input: arg(0),
            out: FuseArg::Output(0, FuseType::F32, LayoutInfo::IsRef),
        })
    }

    fn settings(ref_layout: RefLayoutSetting, choose_output_layout: bool) -> FuseSettings {
        FuseSettings {
            broadcast: true,
            output_shape_updates: true,
            inplace: true,
            vectorization: VectorizationSetting::Activated,
            ref_layout,
            choose_output_layout,
        }
    }

    #[test]
    fn an_elementwise_block_may_choose_its_layout() {
        let settings = settings(RefLayoutSetting::Any, true);

        assert!(may_permute_layout(&settings, &[elemwise_op()]));
        assert!(may_permute_layout(&settings, &[]));
    }

    #[test]
    fn a_runner_that_did_not_opt_in_keeps_the_contiguous_layout() {
        // What a matmul block is: `RefLayoutSetting::Any` by way of
        // `FuseSettings::default()`, but its output view is built from the
        // reference's last two strides while the matmul algorithm is told the output
        // is row-major. A permuted reference makes those two disagree.
        let settings = settings(RefLayoutSetting::Any, false);

        assert!(!may_permute_layout(&settings, &[elemwise_op()]));
        assert!(
            !may_permute_layout(&FuseSettings::default(), &[elemwise_op()]),
            "opting in has to be deliberate, so the default must not",
        );
    }

    #[test]
    fn a_block_that_needs_a_contiguous_reference_may_not_choose() {
        for ref_layout in [
            RefLayoutSetting::OnlyContiguous,
            RefLayoutSetting::SameAsBlock { block_pos: 0 },
        ] {
            assert!(!may_permute_layout(
                &settings(ref_layout, true),
                &[elemwise_op()]
            ));
        }
    }

    #[test]
    fn an_operation_that_indexes_the_last_dimension_rules_out_a_permuted_layout() {
        // These index their operands against `rank - 1` directly rather than through
        // the reference, so they agree with the rest of the kernel only while the
        // reference is contiguous.
        let indexing = [
            FuseOp::Gather {
                input: arg(0),
                indices: arg(1),
                output: FuseArg::Output(0, FuseType::F32, LayoutInfo::IsRef),
                dim: 0,
            },
            FuseOp::Select {
                input: arg(0),
                indices: arg(1),
                output: FuseArg::Output(0, FuseType::F32, LayoutInfo::IsRef),
                dim: 0,
            },
            FuseOp::Cat {
                inputs: vec![arg(0), arg(1)],
                output: FuseArg::Output(0, FuseType::F32, LayoutInfo::IsRef),
                dim: 0,
            },
            FuseOp::Dequantize {
                values: arg(0),
                params: arg(1),
                output: FuseArg::Output(0, FuseType::F32, LayoutInfo::IsRef),
                scheme: QuantSchemeFuse {
                    scheme: QuantScheme::default(),
                },
            },
        ];

        let settings = settings(RefLayoutSetting::Any, true);

        for op in indexing {
            assert!(
                !may_permute_layout(&settings, &[elemwise_op(), op.clone()]),
                "{op:?} must pin the block to a contiguous layout",
            );
        }
    }
}
