use std::{collections::BTreeMap, marker::PhantomData};

use burn_fusion::stream::Context;
use burn_tensor::repr::TensorId;

use crate::{
    fusion::{
        on_write::{ir::ElemwiseOp, settings::FuseSettings},
        JitFusionHandle,
    },
    JitRuntime,
};

use super::{HandleOutput, LaunchPlan, Reshape, TraceRunner};

/// Select the best vectorization factor for each tensor handle.
pub struct VectorizationPlanner<'a, R: JitRuntime> {
    settings: &'a FuseSettings,
    reshapes: &'a Vec<Reshape>,
    reads: &'a BTreeMap<TensorId, Vec<ElemwiseOp>>,
    _r: PhantomData<R>,
}

impl<'a, R: JitRuntime> VectorizationPlanner<'a, R> {
    pub fn new(
        reshapes: &'a Vec<Reshape>,
        reads: &'a BTreeMap<TensorId, Vec<ElemwiseOp>>,
        settings: &'a FuseSettings,
    ) -> Self {
        Self {
            settings,
            reshapes,
            reads,
            _r: PhantomData,
        }
    }
    pub fn run<Runner: TraceRunner<R>>(
        self,
        context: &mut Context<'_, JitFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
    ) {
        let tensors_reshaped = self.reshapes.iter().map(|reshape| {
            (
                context.tensors.get(&reshape.reshaped).unwrap(),
                context.tensors.get(&reshape.original).unwrap(),
                self.reads.get(&reshape.original).unwrap().len() > 1,
            )
        });

        Runner::vectorization(
            &mut plan.vectorization,
            plan.handle_inputs.iter().map(|item| &item.handle),
            plan.global_inputs.iter(),
            plan.global_outputs.iter(),
            tensors_reshaped,
        );

        // If mix vectorization is disable, we set the vectorization factor of each tensor to the
        // minimum value found.
        if !self.settings.mix_vectorization {
            let factor = plan.vectorization.values().min().cloned();
            if let Some(factor) = factor {
                plan.vectorization
                    .iter_mut()
                    .for_each(|(_, vf)| *vf = factor);
            }
        }

        for handle in plan.handle_inputs.iter_mut() {
            handle.vectorization = *plan.vectorization.get(&handle.global_id).unwrap();
        }
        for handle in plan.handle_outputs.iter_mut() {
            if let HandleOutput::Owned {
                vectorization,
                global_id,
                ..
            } = handle
            {
                *vectorization = *plan.vectorization.get(global_id).unwrap()
            }
        }
    }
}
