use super::{BlockPlan, FuseResources, InputReference, TensorView, block::FuseBlock};
use crate::CubeFusionHandle;
use burn_fusion::stream::Context;
use burn_ir::{TensorIr, TensorStatus};
use cubecl::Runtime;
use std::marker::PhantomData;

use super::{HandleInput, LaunchPlan, PotentialInplace};

/// Fetch and register [input handles](HandleInput). Also identifies potential inputs that
/// can be used inplace and/or as the [reference layout](super::super::ir::RefLayout).
pub struct InputPlanner<'a, R: Runtime> {
    resources: &'a FuseResources,
    blocks: &'a Vec<FuseBlock>,
    _r: PhantomData<R>,
}

impl<'a, R: Runtime> InputPlanner<'a, R> {
    pub fn new(resources: &'a FuseResources, blocks: &'a Vec<FuseBlock>) -> Self {
        Self {
            resources,
            blocks,
            _r: PhantomData,
        }
    }

    pub fn run(self, context: &mut Context<'_, CubeFusionHandle<R>>, plan: &mut LaunchPlan<'a, R>) {
        for (pos, (tensor_relative, precision)) in self.resources.inputs.iter().enumerate() {
            let mut tensor_global = context.tensors.get(&tensor_relative.id).unwrap().clone();
            // Important to take the status of the relative graph and not
            // the global graph, since the status of the global graph
            // might be of a later operation on the same tensor id.
            let status = &tensor_relative.status;
            let mut handle = context.handles.get_handle(&tensor_global.id, status);

            self.analyze(plan, pos, tensor_relative, &handle);

            if tensor_global.shape.len() < plan.rank {
                let num_elem: usize = tensor_global.shape.iter().product();
                for _ in 0..(plan.rank - tensor_global.shape.len()) {
                    tensor_global.shape.insert(0, 1);
                    handle.strides.insert(0, num_elem);
                }
            }

            plan.handle_inputs.push(HandleInput {
                precision: *precision,
                handle,
                relative_id: tensor_relative.id,
                global_id: tensor_global.id,
                global_shape: tensor_global.shape.clone(),
                vectorization: 1,
                broadcated: false,
            });
            plan.global_inputs.push(tensor_global);
        }
    }

    fn analyze(
        &self,
        plan: &mut LaunchPlan<'a, R>,
        pos: usize,
        tensor_relative: &'a TensorIr,
        handle: &CubeFusionHandle<R>,
    ) {
        if !self
            .resources
            .inputs_unhandled
            .contains(&tensor_relative.id)
        {
            let mut is_a_view = false;
            // For each view we try to see if it's not possible to set it as a reference input.
            for view in self.resources.views.iter() {
                for (block_plan, block) in plan.blocks.iter_mut().zip(self.blocks) {
                    is_a_view = is_a_view
                        || Self::analyze_view(pos, tensor_relative, block, block_plan, view);
                }
            }

            // Views can't be used inplace.
            if is_a_view {
                return;
            }

            self.analyze_potential_inplace(plan, pos, tensor_relative, handle);
        }
    }

    /// Analyzes if the given tensor can be used inplace in one of the block.
    fn analyze_potential_inplace(
        &self,
        plan: &mut LaunchPlan<'a, R>,
        pos: usize,
        tensor_relative: &'a TensorIr,
        handle: &CubeFusionHandle<R>,
    ) {
        enum BlockInplaceSelection {
            Notinit,
            /// The block reads the input, and therefore can use it for inplace.
            Selected(usize),
            /// The same input is used in multiple blocks.
            Unavailable,
        }

        let mut block_inplace_selection = BlockInplaceSelection::Notinit;

        for (idx, block) in plan.blocks.iter().enumerate() {
            if block.reads.contains_key(&tensor_relative.id) {
                match block_inplace_selection {
                    BlockInplaceSelection::Notinit => {
                        block_inplace_selection = BlockInplaceSelection::Selected(idx);
                    }
                    BlockInplaceSelection::Selected(_) => {
                        block_inplace_selection = BlockInplaceSelection::Unavailable;
                    }
                    BlockInplaceSelection::Unavailable => {}
                }
            }
        }

        if let BlockInplaceSelection::Selected(idx) = block_inplace_selection {
            if self.blocks[idx].shape_ref != tensor_relative.shape {
                return;
            }

            let block_plan = &mut plan.blocks[idx];
            if tensor_relative.status == TensorStatus::ReadWrite
                && handle.handle.can_mut()
                && self.blocks[idx].settings.inplace
            {
                block_plan.potential_inplaces.push(PotentialInplace {
                    input_pos: pos,
                    tensor_relative,
                    strides: handle.strides.clone(),
                });
                // Inplace tensors are normally really good as the reference layout, since
                // it's normally better to be based on writes rather than on reads.
                block_plan.potential_reference_input =
                    Some(InputReference::Normal { input_pos: pos });
            } else {
                block_plan.potential_reference_input =
                    Some(InputReference::Normal { input_pos: pos });
            }
        }
    }

    /// Analyzes if the given tensor is also the view provided, and check if it can be used as the reference layout
    /// for the given block.
    fn analyze_view(
        pos: usize,
        tensor_relative: &'a TensorIr,
        block: &FuseBlock,
        block_plan: &mut BlockPlan<'a>,
        view: &TensorView,
    ) -> bool {
        match view {
            TensorView::Reshape {
                reshaped,
                original,
                reshape_pos,
                shape_relative,
            } => {
                if original == &tensor_relative.id || reshaped == &tensor_relative.id {
                    if block_plan.potential_reference_input.is_none()
                        && shape_relative == &block.shape_ref
                    {
                        block_plan.potential_reference_input = Some(InputReference::Reshaped {
                            reshape_pos: *reshape_pos as usize,
                        });
                    }
                    return true;
                }
            }
            TensorView::SwapDims {
                swapped,
                original,
                dims,
                ..
            } => {
                if swapped == &tensor_relative.id {
                    return true;
                }

                if original == &tensor_relative.id {
                    let mut shape = tensor_relative.shape.clone();
                    shape.swap(dims.0 as usize, dims.1 as usize);

                    if block_plan.potential_reference_input.is_none() && shape == block.shape_ref {
                        block_plan.potential_reference_input = Some(InputReference::SwapDims {
                            original_pos: pos,
                            dims: *dims,
                        });
                    }
                    return true;
                }
            }
        };

        false
    }
}
