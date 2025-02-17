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
        let mut handles = Vec::with_capacity(self.inputs.len());
        let mut globals = Vec::with_capacity(self.inputs.len());

        for _ in 0..self.inputs.len() {
            handles.push(None);
            globals.push(None);
        }

        for (precision, (pos, tensor_relative)) in self.inputs.iter() {
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
                    input_pos: *pos as usize,
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

            handles[*pos as usize] = Some(HandleInput {
                precision,
                handle,
                relative_id: tensor_relative.id,
                global_id: tensor_global.id,
                global_shape: tensor_global.shape.clone(),
                vectorization: 1,
            });
            globals[*pos as usize] = Some(tensor_global);
        }

        for (handle, global) in handles.into_iter().zip(globals.into_iter()) {
            plan.handle_inputs.push(handle.unwrap());
            plan.global_inputs.push(global.unwrap());
        }
    }
}
