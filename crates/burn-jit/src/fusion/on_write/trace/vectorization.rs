use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
};

use burn_fusion::stream::Context;
use burn_ir::TensorId;
use cubecl::tensor_line_size_parallel;

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
    indexed: &'a BTreeSet<TensorId>,
    line_size_overrides: &'a Vec<TensorId>,
    _r: PhantomData<R>,
}

impl<'a, R: JitRuntime> VectorizationPlanner<'a, R> {
    pub fn new(
        reshapes: &'a Vec<Reshape>,
        reads: &'a BTreeMap<TensorId, Vec<ElemwiseOp>>,
        settings: &'a FuseSettings,
        indexed: &'a BTreeSet<TensorId>,
        line_size_overrides: &'a Vec<TensorId>,
    ) -> Self {
        Self {
            settings,
            reshapes,
            reads,
            indexed,
            line_size_overrides,
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

        let filtered = plan
            .handle_inputs
            .iter()
            .map(|item| !self.indexed.contains(&item.relative_id))
            .collect::<Vec<_>>();

        Runner::vectorization(
            &mut plan.vectorization,
            plan.handle_inputs
                .iter()
                .enumerate()
                .filter_map(|(i, item)| {
                    if filtered[i] {
                        Some(&item.handle)
                    } else {
                        None
                    }
                }),
            plan.global_inputs
                .iter()
                .enumerate()
                .filter_map(|(i, item)| if filtered[i] { Some(item) } else { None }),
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

        for tensor in self.indexed {
            let global = context.tensors.get(tensor).unwrap();
            plan.vectorization.insert(global.id, 1);
        }

        for handle in plan.handle_inputs.iter_mut() {
            if self.line_size_overrides.contains(&handle.relative_id)
                && !self.reads.contains_key(&handle.relative_id)
            {
                let elem = handle.precision.into();
                let line_size = tensor_line_size_parallel(
                    R::line_size_elem(&elem),
                    &handle.global_shape,
                    &handle.handle.strides,
                    handle.handle.strides.len() - 1,
                );
                handle.vectorization = line_size;
            } else {
                handle.vectorization = match plan.vectorization.get(&handle.global_id) {
                    Some(v) => *v,
                    None => panic!("No vectorization factor found for {:?}", handle.global_id),
                };
            }
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
