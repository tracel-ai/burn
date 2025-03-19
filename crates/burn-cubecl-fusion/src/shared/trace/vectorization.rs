use std::marker::PhantomData;

use burn_fusion::stream::Context;
use burn_ir::TensorId;
use cubecl::{
    ir::{Elem, UIntKind},
    Runtime,
};

use crate::{shared::trace::Vect, CubeFusionHandle};

use super::{HandleOutput, KernelResources, LaunchPlan, TensorView, Vectorization};

/// Select the best vectorization factor for each tensor handle.
pub struct VectorizationPlanner<'a, R: Runtime> {
    resources: &'a KernelResources,
    _r: PhantomData<R>,
}

impl<'a, R: Runtime> VectorizationPlanner<'a, R> {
    pub fn new(resources: &'a KernelResources) -> Self {
        Self {
            resources,
            _r: PhantomData,
        }
    }
    pub fn run<Runner: Vectorization<R>>(
        self,
        runner: &Runner,
        context: &Context<'_, CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
    ) {
        let has_multiple_read = |tensor: &TensorId| {
            let mut read_count = 0;
            for block in plan.blocks.iter() {
                read_count += block.reads.get(tensor).map(|a| a.len()).unwrap_or(0);
            }
            read_count > 1
        };
        let tensors_reshaped = self.resources.views.iter().filter_map(|view| match view {
            TensorView::Reshape {
                reshaped, original, ..
            } => Some((
                context.tensors.get(reshaped).unwrap(),
                context.tensors.get(original).unwrap(),
                has_multiple_read(original),
            )),
            TensorView::SwapDims { .. } => None,
        });
        let tensors_swapped = self.resources.views.iter().filter_map(|view| match view {
            TensorView::SwapDims {
                swapped,
                original,
                dims,
                ..
            } => Some((
                context.tensors.get(swapped).unwrap(),
                context.tensors.get(original).unwrap(),
                has_multiple_read(original),
                dims,
            )),
            TensorView::Reshape { .. } => None,
        });

        let mut ref_elem = (Elem::UInt(UIntKind::U64), 8);

        for r in plan.global_inputs.iter() {
            let elem: Elem = r.dtype.into();
            let elem_size = elem.size();

            if ref_elem.1 >= elem_size {
                ref_elem = (elem, elem_size);
            }
        }
        for r in plan.global_outputs.iter() {
            let elem: Elem = r.dtype.into();
            let elem_size = elem.size();

            if ref_elem.1 >= elem_size {
                ref_elem = (elem, elem_size);
            }
        }

        let filtered = plan
            .handle_inputs
            .iter()
            .map(|item| !self.resources.indexed.contains_key(&item.relative_id))
            .collect::<Vec<_>>();

        Runner::vectorization(
            &mut plan.vectorizations,
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
            tensors_swapped,
            &ref_elem.0,
            u8::MAX,
            runner.axis(),
        );

        for tensor in self.resources.indexed.keys() {
            let global = context.tensors.get(tensor).unwrap();
            plan.vectorizations.insert(global.id, Vect::Aligned(1));
        }

        for handle in plan.handle_inputs.iter_mut() {
            let (vect, br) = match plan.vectorizations.get(&handle.global_id) {
                Some(v) => (v.line_size(), v.is_broadcast()),
                None => panic!("No vectorization factor found for {:?}", handle.global_id),
            };
            handle.vectorization = vect;
            handle.broadcated = br;

            for block in plan.blocks.iter_mut() {
                if block.reads.contains_key(&handle.relative_id)
                    && block.width < handle.vectorization
                {
                    block.width = handle.vectorization;
                }
            }
        }

        for handle in plan.handle_outputs.iter_mut() {
            if let HandleOutput::Owned {
                vectorization,
                global_id,
                relative_id,
                ..
            } = handle
            {
                *vectorization = plan.vectorizations.get(global_id).unwrap().line_size();

                for block in plan.blocks.iter_mut() {
                    if block.writes.contains_key(&relative_id) && block.width < *vectorization {
                        block.width = *vectorization;
                    }
                }
            }
        }
    }
}
