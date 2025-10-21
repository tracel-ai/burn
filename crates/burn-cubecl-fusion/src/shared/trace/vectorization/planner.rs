use std::marker::PhantomData;

use burn_fusion::stream::Context;
use burn_ir::TensorId;
use cubecl::{
    Runtime,
    ir::{ElemType, StorageType, UIntKind},
};
use cubecl_quant::scheme::{QuantScheme, QuantStore, QuantValue};

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

        let mut ref_elem = (ElemType::UInt(UIntKind::U64).into(), 8);
        let mut quants_line_sizes: Option<Vec<u8>> = None;

        for input in plan.handle_inputs.iter() {
            let elem: StorageType = match input {
                HandleInput::Normal(h) => h.global_ir.dtype.into(),
                HandleInput::QuantValues(handle) => match handle.global_ir.dtype {
                    burn_tensor::DType::QFloat(scheme) => {
                        line_sizes_quants::<R>(&mut quants_line_sizes, scheme);
                        continue;
                    }
                    _ => panic!("Unable to retrieve the scheme for quantized values."),
                },
                HandleInput::QuantParams(..) => continue,
            };
            let elem_size = elem.size();

            if ref_elem.1 >= elem_size {
                ref_elem = (elem, elem_size);
            }
        }
        for r in plan.global_outputs.iter() {
            let elem: StorageType = r.dtype.into();
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
                    // Filter out indexed resources.
                    .map(|item| !self.resources.indexed.contains_key(&item.relative_id))
                    .unwrap_or(true)
            })
            .collect::<Vec<_>>();

        let line_sizes = match quants_line_sizes {
            // Quantization normally triggers higher vectorization than anything else, no need to
            // compare to ref elem.
            Some(line_sizes) => line_sizes,
            None => R::io_optimized_line_sizes_unchecked(&ref_elem.0).collect::<Vec<u8>>(),
        };

        let vectorization_axis = runner.axis(plan);

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
                            HandleInput::QuantValues(h) => {
                                VectorizationHandle::QuantValues(&h.handle, &h.global_ir)
                            }
                            HandleInput::QuantParams(h) => {
                                VectorizationHandle::QuantParams(&h.handle)
                            }
                        })
                    } else {
                        None
                    }
                }),
            plan.global_outputs.iter(),
            tensors_reshaped,
            tensors_swapped,
            &line_sizes,
            u8::MAX,
            vectorization_axis,
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
                HandleInput::QuantValues(h) => (&h.global_ir, &h.relative_id),
                HandleInput::QuantParams(_) => continue,
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

#[derive(Debug)]
enum VectorizationAction {
    Input(usize),
    Output(usize),
}

#[derive(Debug)]
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
                    HandleInput::QuantValues(input) => {
                        input.vectorization = vect;
                    }
                    HandleInput::QuantParams(_) => {
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

fn line_sizes_quants<R: Runtime>(quants_line_sizes: &mut Option<Vec<u8>>, scheme: QuantScheme) {
    match scheme.store {
        QuantStore::Native => match scheme.value {
            // Type sizes are the same so just treat fp8/fp4x2 as i8
            QuantValue::Q8F
            | QuantValue::Q8S
            | QuantValue::E4M3
            | QuantValue::E5M2
            | QuantValue::E2M1 => {
                let line_sizes = R::io_optimized_line_sizes_unchecked(
                    &ElemType::Int(cubecl::ir::IntKind::I8).into(),
                )
                .collect::<Vec<u8>>();

                match &quants_line_sizes {
                    Some(sizes) => {
                        if sizes[0] < line_sizes[0] {
                            *quants_line_sizes = Some(line_sizes);
                        }
                    }
                    None => {
                        *quants_line_sizes = Some(line_sizes);
                    }
                }
            }
            QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => {
                unreachable!("Can't store native sub-byte values")
            }
        },
        QuantStore::U32 => {
            let mut line_sizes = R::io_optimized_line_sizes_unchecked(
                &ElemType::Int(cubecl::ir::IntKind::I32).into(),
            )
            .collect::<Vec<u8>>();
            for val in line_sizes.iter_mut() {
                *val *= scheme.num_quants() as u8;
            }

            match &quants_line_sizes {
                Some(sizes) => {
                    if sizes[0] < line_sizes[0] {
                        let mut min = *line_sizes.last().unwrap();

                        while min > 1 {
                            min /= 2;
                            line_sizes.push(min);
                        }
                        *quants_line_sizes = Some(line_sizes);
                    }
                }
                None => {
                    *quants_line_sizes = Some(line_sizes);
                }
            }
        }
    };
}
