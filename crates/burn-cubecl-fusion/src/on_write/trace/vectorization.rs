use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
};

use burn_fusion::stream::Context;
use burn_ir::TensorId;
use cubecl::{
    ir::{Elem, UIntKind},
    Runtime,
};

use crate::{
    on_write::{ir::ElemwiseOp, trace::Vect},
    CubeFusionHandle,
};

use super::{HandleOutput, LaunchPlan, TensorView, TraceRunner};

/// Select the best vectorization factor for each tensor handle.
pub struct VectorizationPlanner<'a, R: Runtime> {
    views: &'a Vec<TensorView>,
    reads: &'a BTreeMap<TensorId, Vec<ElemwiseOp>>,
    indexed: &'a BTreeSet<TensorId>,
    _r: PhantomData<R>,
}

impl<'a, R: Runtime> VectorizationPlanner<'a, R> {
    pub fn new(
        views: &'a Vec<TensorView>,
        reads: &'a BTreeMap<TensorId, Vec<ElemwiseOp>>,
        indexed: &'a BTreeSet<TensorId>,
    ) -> Self {
        Self {
            views,
            reads,
            indexed,
            _r: PhantomData,
        }
    }
    pub fn run<Runner: TraceRunner<R>>(
        self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
    ) {
        let tensors_reshaped = self.views.iter().filter_map(|view| match view {
            TensorView::Reshape { reshaped, original } => Some((
                context.tensors.get(reshaped).unwrap(),
                context.tensors.get(original).unwrap(),
                self.reads.get(original).unwrap().len() > 1,
            )),
            TensorView::SwapDims { .. } => None,
        });
        let tensors_swapped = self.views.iter().filter_map(|view| match view {
            TensorView::SwapDims {
                swapped,
                original,
                dims,
            } => Some((
                context.tensors.get(swapped).unwrap(),
                context.tensors.get(original).unwrap(),
                self.reads.get(original).unwrap().len() > 1,
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
            tensors_swapped,
            &ref_elem.0,
        );

        for tensor in self.indexed {
            let global = context.tensors.get(tensor).unwrap();
            plan.vectorization.insert(global.id, Vect::Aligned(1));
        }

        plan.width = 0;

        for handle in plan.handle_inputs.iter_mut() {
            let (vect, br) = match plan.vectorization.get(&handle.global_id) {
                Some(v) => (v.line_size(), v.is_broadcast()),
                None => panic!("No vectorization factor found for {:?}", handle.global_id),
            };
            handle.vectorization = vect;
            handle.broadcated = br;
            if plan.width < handle.vectorization {
                plan.width = handle.vectorization;
            }
        }
        for handle in plan.handle_outputs.iter_mut() {
            if let HandleOutput::Owned {
                vectorization,
                global_id,
                ..
            } = handle
            {
                *vectorization = plan.vectorization.get(global_id).unwrap().line_size();
                if plan.width < *vectorization {
                    plan.width = *vectorization;
                }
            }
        }
    }
}
