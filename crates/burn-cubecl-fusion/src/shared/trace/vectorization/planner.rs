use std::marker::PhantomData;

use burn_fusion::stream::Context;
use burn_ir::TensorId;
use cubecl::{
    Runtime,
    ir::{Elem, UIntKind},
};

use crate::{
    CubeFusionHandle,
    shared::{
        settings::VectorizationSetting,
        trace::{HandleInput, VectorizationHandle},
    },
};

use super::{
    super::{
        BlockPlan, FuseResources, HandleOutput, LaunchPlan, TensorView, Vectorization,
        block::FuseBlock,
    },
    Vect,
};

/// Select the best vectorization factor for each tensor handle.
pub struct VectorizationPlanner<'a, R: Runtime> {
    resources: &'a FuseResources,
    blocks: &'a Vec<FuseBlock>,
    _r: PhantomData<R>,
}

impl<'a, R: Runtime> VectorizationPlanner<'a, R> {
    pub fn new(resources: &'a FuseResources, blocks: &'a Vec<FuseBlock>) -> Self {
        Self {
            resources,
            blocks,
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

        for input in plan.handle_inputs.iter() {
            let elem: Elem = match input {
                HandleInput::Normal(h) => h.global_ir.dtype.into(),
                HandleInput::QuantData(..) => continue,
                HandleInput::QuantScales(..) => continue,
            };
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
            .map(|item| {
                item.as_normal()
                    // Filter out indexed ressources.
                    .map(|item| !self.resources.indexed.contains_key(&item.relative_id))
                    .unwrap_or(true)
            })
            .collect::<Vec<_>>();

        runner.vectorization(
            context,
            &mut plan.vectorizations,
            plan.handle_inputs
                .iter()
                .enumerate()
                .filter_map(|(i, item)| {
                    if filtered[i] {
                        Some(match item {
                            HandleInput::Normal(h) => {
                                VectorizationHandle::NormalInput(&h.handle, &h.global_ir)
                            }
                            HandleInput::QuantData(h) => {
                                VectorizationHandle::QuantData(&h.handle, &h.global_ir)
                            }
                            HandleInput::QuantScales(h) => {
                                VectorizationHandle::QuantScales(&h.handle)
                            }
                        })
                    } else {
                        None
                    }
                }),
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

        let mut block_vectorization = Vec::with_capacity(self.blocks.len());
        for _ in 0..self.blocks.len() {
            block_vectorization.push(Vec::new());
        }

        for (input_pos, handle) in plan.handle_inputs.iter_mut().enumerate() {
            let (global_ir, relative_id) = match handle {
                HandleInput::Normal(h) => (&h.global_ir, &h.relative_id),
                HandleInput::QuantData(h) => (&h.global_ir, &h.relative_id),
                HandleInput::QuantScales(_) => continue,
            };
            let (vect, br) = match plan.vectorizations.get(&global_ir.id) {
                Some(v) => (v.line_size(), v.is_broadcast()),
                None => panic!("No vectorization factor found for {:?}", global_ir.id),
            };

            for (block_pos, block_plan) in plan.blocks.iter().enumerate() {
                if block_plan.reads.contains_key(relative_id) {
                    block_vectorization[block_pos].push(BlockVectorization {
                        action: VectorizationAction::Input(input_pos),
                        potential: vect,
                        broadcasted: br,
                    });
                }
            }
        }

        for (output_pos, handle) in plan.handle_outputs.iter().enumerate() {
            if let HandleOutput::Owned {
                global_id,
                relative_id,
                ..
            } = handle
            {
                for (block_pos, block_plan) in plan.blocks.iter().enumerate() {
                    if block_plan.writes.contains_key(relative_id) {
                        let vectorization = plan.vectorizations.get(global_id).unwrap().line_size();
                        block_vectorization[block_pos].push(BlockVectorization {
                            action: VectorizationAction::Output(output_pos),
                            potential: vectorization,
                            broadcasted: false,
                        });
                    }
                }
            }
        }

        let mut previous_width = 1;

        for ((tmp, block_plan), block) in block_vectorization
            .into_iter()
            .zip(plan.blocks.iter_mut())
            .zip(self.blocks)
        {
            match block.settings.vectorization {
                VectorizationSetting::Activated => {
                    apply_vectorization_block(
                        tmp,
                        &mut plan.handle_inputs,
                        &mut plan.handle_outputs,
                        block_plan,
                        u8::MAX,
                    );
                }
                VectorizationSetting::SmallerOrEqualThanPreviousBlock => {
                    apply_vectorization_block(
                        tmp,
                        &mut plan.handle_inputs,
                        &mut plan.handle_outputs,
                        block_plan,
                        previous_width,
                    );
                }
                VectorizationSetting::Deactivated => {
                    apply_vectorization_block(
                        tmp,
                        &mut plan.handle_inputs,
                        &mut plan.handle_outputs,
                        block_plan,
                        1,
                    );
                }
            }
            previous_width = block_plan.width;
        }
    }
}

enum VectorizationAction {
    Input(usize),
    Output(usize),
}

struct BlockVectorization {
    action: VectorizationAction,
    potential: u8,
    broadcasted: bool,
}

fn apply_vectorization_block<R: Runtime>(
    block_vectorization: Vec<BlockVectorization>,
    inputs: &mut [HandleInput<R>],
    outputs: &mut [HandleOutput<R>],
    block_plan: &mut BlockPlan,
    max: u8,
) {
    for item in block_vectorization {
        match item.action {
            VectorizationAction::Input(pos) => {
                let (vect, br) = if item.potential <= max {
                    (item.potential, item.broadcasted)
                } else {
                    (1, false)
                };

                match &mut inputs[pos] {
                    HandleInput::Normal(input) => {
                        input.vectorization = vect;
                        input.broadcated = br;
                    }
                    HandleInput::QuantData(input) => {
                        input.vectorization = vect;
                    }
                    HandleInput::QuantScales(_) => {
                        // Not vectorized
                    }
                }

                if block_plan.width < vect {
                    block_plan.width = vect;
                }
            }
            VectorizationAction::Output(pos) => {
                if let HandleOutput::Owned { vectorization, .. } = &mut outputs[pos] {
                    let vect = if item.potential <= max {
                        item.potential
                    } else {
                        1
                    };
                    *vectorization = vect;

                    if block_plan.width < vect {
                        block_plan.width = vect;
                    }
                }
            }
        }
    }
}
