use super::TensorView;
use crate::{on_write::settings::FuseSettings, CubeFusionHandle};
use burn_fusion::stream::Context;
use burn_ir::{TensorId, TensorStatus};
use cubecl::Runtime;
use std::marker::PhantomData;

use super::{HandleInput, LaunchPlan, PotentialInplace, RegisteredTensors};

/// Fetch and register [input handles](HandleInput) and itendify potential inputs that
/// can be used inplace.
pub struct InputPlanner<'a, R: Runtime> {
    inputs: &'a RegisteredTensors,
    inputs_unhandled: &'a Vec<TensorId>,
    views: &'a Vec<TensorView>,
    shape_ref: &'a Vec<usize>,
    settings: &'a FuseSettings,
    _r: PhantomData<R>,
}

impl<'a, R: Runtime> InputPlanner<'a, R> {
    pub fn new(
        inputs: &'a RegisteredTensors,
        inputs_unhandled: &'a Vec<TensorId>,
        views: &'a Vec<TensorView>,
        shape_ref: &'a Vec<usize>,
        settings: &'a FuseSettings,
    ) -> Self {
        Self {
            inputs,
            settings,
            inputs_unhandled,
            views,
            shape_ref,
            _r: PhantomData,
        }
    }

    pub fn run(self, context: &mut Context<'_, CubeFusionHandle<R>>, plan: &mut LaunchPlan<'a, R>) {
        for (pos, (tensor_relative, precision)) in self.inputs.iter().enumerate() {
            let mut tensor_global = context.tensors.get(&tensor_relative.id).unwrap().clone();
            // Important to take the status of the relative graph and not
            // the global graph, since the status of the global graph
            // might be of a later operation on the same tensor id.
            let status = &tensor_relative.status;
            let mut handle = context.handles.get_handle(&tensor_global.id, status);

            if self.settings.inplace
                && status == &TensorStatus::ReadWrite
                && handle.handle.can_mut()
                && !self.inputs_unhandled.contains(&tensor_relative.id)
                && !self.views.iter().any(|v| match v {
                    TensorView::Reshape { reshaped, original } => {
                        reshaped == &tensor_relative.id || original == &tensor_relative.id
                    }
                    TensorView::SwapDims {
                        swapped, original, ..
                    } => swapped == &tensor_relative.id || original == &tensor_relative.id,
                })
                && self.shape_ref == &tensor_relative.shape
            {
                plan.potential_inplaces.push(PotentialInplace {
                    input_pos: pos,
                    tensor_relative,
                    strides: handle.strides.clone(),
                });
            }

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
}
