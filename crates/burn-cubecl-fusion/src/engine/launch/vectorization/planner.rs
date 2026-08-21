use super::{
    super::{BlockPlan, HandleOutput, LaunchPlan},
    Vect,
};
use crate::{
    CubeFusionHandle,
    engine::{
        launch::{
            HandleInput, ReferenceSelection,
            layout::permuted_innermost_dim,
            runner::{Vectorization, VectorizationAxis, VectorizationHandle},
        },
        settings::VectorizationSetting,
        trace::{FuseResources, TensorView, block::FuseBlock},
    },
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_fusion::stream::Context;
use burn_ir::TensorId;
use cubecl::{
    Runtime,
    client::ComputeClient,
    ir::{ElemType, UIntKind},
};
use cubecl::{
    ir::VectorSize,
    quant::scheme::{QuantScheme, QuantStore, QuantValue},
};
use std::collections::HashMap;
use std::marker::PhantomData;

/// What one block asks of one tensor's vectorization axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AxisVote {
    /// The block iterates in logical dimension order, so the tensor keeps the
    /// default axis — its own last dimension.
    Default,
    /// The block iterates in a permuted order innermost along this dimension, and
    /// every tensor it touches has to be measured along the same one.
    Along(usize),
    /// The tensor cannot be lined up with the order the block iterates in, and must
    /// not be vectorized.
    Never,
}

/// The dimension a permuted reference advances a line along, or `None` when the
/// block iterates in logical dimension order.
///
/// A virtual reference indexes through a transform and those paths were written
/// against the last dimension; a contiguous concrete one advances along the last
/// dimension by construction. Neither asks anything of the tensors it touches
/// beyond the default, so both answer `None`.
fn permuted_innermost(reference: &ReferenceSelection) -> Option<usize> {
    let ReferenceSelection::Concrete { shape, strides, .. } = reference else {
        return None;
    };

    permuted_innermost_dim(shape, strides)
}

/// What a block whose reference is innermost along `permuted` asks of one tensor.
///
/// `strides` is given only for tensors the vectorization pass will not check them
/// for itself, which is what makes the difference between refusing and trusting it
/// to refuse.
fn axis_vote(permuted: Option<usize>, rank: usize, strides: Option<&[usize]>) -> AxisVote {
    let Some(axis) = permuted else {
        return AxisVote::Default;
    };

    // A reshaped view can have fewer dimensions than the reference. There is no
    // dimension of its own to line up with the reference's, and reading its shape
    // at the reference's innermost index would index past the end.
    if axis >= rank {
        return AxisVote::Never;
    }

    // A line along a dimension the tensor is not contiguous in covers elements that
    // are not adjacent in its buffer, so it is not a line at all.
    if let Some(strides) = strides
        && strides[axis] != 1
    {
        return AxisVote::Never;
    }

    AxisVote::Along(axis)
}

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

    /// Which dimension of each tensor a vectorized access runs along, and which
    /// tensors must not be vectorized at all.
    ///
    /// Vectorization assumes the last dimension is the innermost in memory, which
    /// holds only for a layout in logical dimension order. A block whose reference
    /// is permuted — what an elementwise block adopts when its inputs come out of a
    /// convolution — has its innermost dimension somewhere else, and measuring the
    /// last one there finds a stride that is not one and gives up on an access that
    /// is perfectly linear.
    ///
    /// Worse than giving up: a tensor whose dimension order differs from the
    /// reference's must not vectorize at all, because a line of the reference and a
    /// line of that tensor cover different elements. Handing every tensor the
    /// reference's innermost dimension enforces that for inputs, since
    /// [`vectorization_input`](super::base) refuses an axis it is not contiguous
    /// along. It does not enforce it for outputs —
    /// [`vectorization_output`](super::base) is given a shape and never sees a
    /// stride — nor for a reshaped view whose rank differs from the reference's, so
    /// those are refused here instead.
    ///
    /// A block iterating in logical dimension order asks for nothing beyond the
    /// default and votes [AxisVote::Default], which leaves every such block behaving
    /// exactly as it did before a block could choose its own layout. Two blocks that
    /// want different things from a shared tensor cancel out to [AxisVote::Never]:
    /// the disagreement means at least one of them would be vectorizing against a
    /// layout that is not its own.
    fn vectorization_axis<Runner: Vectorization<R>>(
        &self,
        runner: &Runner,
        context: &Context<CubeFusionHandle<R>>,
        plan: &LaunchPlan<'a, R>,
    ) -> (VectorizationAxis, Vec<TensorId>) {
        // Keyed by the id the vectorization pass looks the axis up by, carrying the
        // id whose vector size a refusal has to be applied to. Those differ for a
        // view: it is vectorized under its own id, but the verdict lands on the
        // tensor it views.
        let mut per_tensor: HashMap<TensorId, (AxisVote, TensorId)> = HashMap::new();

        let mut vote = |id: TensorId, clamp: TensorId, cast: AxisVote| {
            per_tensor
                .entry(id)
                .and_modify(|(current, _)| {
                    if *current != cast {
                        *current = AxisVote::Never;
                    }
                })
                .or_insert((cast, clamp));
        };

        for block_plan in plan.blocks.iter() {
            let permuted = permuted_innermost(&block_plan.reference);

            for input in plan.handle_inputs.iter() {
                if let Some(input) = input.as_normal()
                    && block_plan.reads.contains_key(&input.relative_id)
                {
                    // No strides: `vectorization_input` checks them itself, and it
                    // tells a broadcast dimension from one it merely cannot line up.
                    let rank = input.global_ir.shape.rank();
                    vote(
                        input.global_ir.id,
                        input.global_ir.id,
                        axis_vote(permuted, rank, None),
                    );
                }
            }

            for output in plan.handle_outputs.iter() {
                if let HandleOutput::Owned {
                    handle,
                    global_id,
                    relative_id,
                    global_shape,
                    ..
                } = output
                    && block_plan.writes.contains_key(relative_id)
                {
                    let rank = global_shape.rank();
                    vote(
                        *global_id,
                        *global_id,
                        axis_vote(permuted, rank, Some(&handle.strides)),
                    );
                }
            }
        }

        // A view is read under its own shape and rank. `vectorization_reshape`
        // indexes the reshaped shape at the axis and bails out unless the axis is
        // that shape's last dimension; `vectorization_swapped` maps the axis through
        // the swap and checks the original's stride there — the same mapping
        // [read_input_aligned](crate::engine::codegen::io) applies. Both of them go
        // wrong only when handed the default while the block iterates elsewhere.
        for view in self.resources.views.iter() {
            let (view_id, original_id) = match view {
                TensorView::Reshape {
                    reshaped, original, ..
                } => (reshaped, original),
                TensorView::SwapDims {
                    swapped, original, ..
                } => (swapped, original),
                // Already pinned to a vector size of one by [Self::run].
                TensorView::NhwcStrides { .. } => continue,
            };

            let (Some(view_global), Some(original_global)) = (
                context.tensors.get(view_id),
                context.tensors.get(original_id),
            ) else {
                continue;
            };

            for block_plan in plan.blocks.iter() {
                if !block_plan.reads.contains_key(original_id) {
                    continue;
                }

                let permuted = permuted_innermost(&block_plan.reference);
                let rank = view_global.shape.rank();

                vote(
                    view_global.id,
                    original_global.id,
                    axis_vote(permuted, rank, None),
                );
            }
        }

        // The runner knows better for its own operands — the matmul one places the
        // axis by matrix layout — so anything it sets wins, refusals included.
        const UNSET: usize = usize::MAX;
        let mut axis = runner.axis(plan);
        let mut never = Vec::new();

        for (id, (cast, clamp)) in per_tensor {
            if axis.get(id, || UNSET) != UNSET {
                continue;
            }

            match cast {
                AxisVote::Default => {}
                AxisVote::Along(dim) => axis.insert(id, dim),
                AxisVote::Never => never.push(clamp),
            }
        }

        (axis, never)
    }

    pub fn run<Runner: Vectorization<R>>(
        self,
        client: &ComputeClient<R>,
        runner: &Runner,
        context: &Context<CubeFusionHandle<R>>,
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
            TensorView::NhwcStrides { .. } => None,
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
            TensorView::NhwcStrides { .. } => None,
        });

        let mut ref_elem = (ElemType::UInt(UIntKind::U64), 8);
        let mut quants_vector_sizes: Option<Vec<VectorSize>> = None;

        for input in plan.handle_inputs.iter() {
            let elem: ElemType = match input {
                HandleInput::Normal(h) => dtype_to_storage_type(h.global_ir.dtype),
                HandleInput::QuantValues(handle) => match handle.global_ir.dtype {
                    burn_std::DType::QFloat(scheme) => {
                        vector_sizes_quants(client, &mut quants_vector_sizes, scheme);
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
            let elem: ElemType = dtype_to_storage_type(r.dtype);
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

        let vector_sizes = match quants_vector_sizes {
            // Quantization normally triggers higher vectorization than anything else, no need to
            // compare to ref elem.
            Some(vector_sizes) => vector_sizes,
            None => client
                .io_optimized_vector_sizes(ref_elem.0.size())
                .collect::<Vec<_>>(),
        };
        let (vectorization_axis, never_vectorized) = self.vectorization_axis(runner, context, plan);

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
                            HandleInput::QuantParams(_) => VectorizationHandle::QuantParams,
                        })
                    } else {
                        None
                    }
                }),
            plan.global_outputs.iter(),
            tensors_reshaped,
            tensors_swapped,
            &vector_sizes,
            u8::MAX as usize,
            vectorization_axis,
        );

        for tensor in self.resources.indexed.keys() {
            let global = context.tensors.get(tensor).unwrap();
            plan.vectorizations.insert(global.id, Vect::Aligned(1));
        }

        for view in self.resources.views.iter() {
            if let TensorView::NhwcStrides { id, .. } = view {
                let global = context.tensors.get(id).unwrap();
                plan.vectorizations.insert(global.id, Vect::Aligned(1));
            }
        }

        // Tensors whose own layout cannot be lined up with the one their block
        // iterates in. A tensor already judged broadcast along its axis is left
        // alone: it is read element by element either way, and calling it aligned
        // would drag the whole block's width down to one with it.
        for id in never_vectorized {
            if !matches!(plan.vectorizations.get(&id), Some(Vect::Broadcasted)) {
                plan.vectorizations.insert(id, Vect::Aligned(1));
            }
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
                Some(v) => (v.vector_size(), v.is_broadcast()),
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
                        let vectorization =
                            plan.vectorizations.get(global_id).unwrap().vector_size();
                        block_vectorization[block_pos].push(BlockVectorization {
                            action: VectorizationAction::Output(output_pos),
                            potential: vectorization,
                            broadcasted: false,
                        });
                    }
                }
            }
        }

        let mut previous_widths = Vec::with_capacity(block_vectorization.len());

        // Unhandled inputs might not get included in any fused blocks for now.
        //
        // So we ensure they are vectorized by setting their vectorization before we set the
        // vectorizations in blocks.
        //
        // Unhandled Outputs are correctly vectorized, so this is only necessary for inputs.
        for input in self.resources.inputs_unhandled.iter() {
            let pos = self
                .resources
                .inputs
                .get_index(*input)
                .unwrap_or_else(|| self.resources.inputs.get_index_quant(*input).unwrap());
            let input_global = context.tensors.get(input).unwrap();

            match plan.vectorizations.get(&input_global.id).unwrap() {
                Vect::Aligned(vect) => {
                    let handle = &mut plan.handle_inputs[pos];
                    match handle {
                        HandleInput::Normal(handle) => {
                            handle.vector_size = *vect;
                        }
                        HandleInput::QuantValues(handle) => {
                            handle.vector_size = *vect;
                        }
                        HandleInput::QuantParams(_) => {}
                    }
                }
                Vect::Broadcasted => {}
            }
        }

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
                        u8::MAX as usize,
                    );
                }
                VectorizationSetting::SmallerOrEqualThanPreviousBlock { block_pos } => {
                    apply_vectorization_block(
                        tmp,
                        &mut plan.handle_inputs,
                        &mut plan.handle_outputs,
                        block_plan,
                        previous_widths[block_pos],
                    );
                    if block_plan.width == 0 {
                        block_plan.width = previous_widths[block_pos];
                    }
                }
                VectorizationSetting::EqualThanPreviousBlock { block_pos } => {
                    apply_vectorization_block(
                        tmp,
                        &mut plan.handle_inputs,
                        &mut plan.handle_outputs,
                        block_plan,
                        previous_widths[block_pos],
                    );
                    // Enforces the width.
                    block_plan.width = previous_widths[block_pos];
                }
                VectorizationSetting::Deactivated => {
                    apply_vectorization_block(
                        tmp,
                        &mut plan.handle_inputs,
                        &mut plan.handle_outputs,
                        block_plan,
                        1,
                    );
                    block_plan.width = 1;
                }
            }

            // When only virtual inputs/outputs are present for a block, we need to set a width.
            if block_plan.width == 0 {
                if let Some(w) = previous_widths.last() {
                    block_plan.width = *w;
                } else {
                    block_plan.width = 1;
                }
            }

            previous_widths.push(block_plan.width);
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
    potential: VectorSize,
    broadcasted: bool,
}

fn apply_vectorization_block<R: Runtime>(
    block_vectorization: Vec<BlockVectorization>,
    inputs: &mut [HandleInput<R>],
    outputs: &mut [HandleOutput<R>],
    block_plan: &mut BlockPlan,
    max: VectorSize,
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
                        input.vector_size = vect;
                        input.broadcated = br;
                    }
                    HandleInput::QuantValues(input) => {
                        input.vector_size = vect;
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

fn vector_sizes_quants<R: Runtime>(
    client: &ComputeClient<R>,
    quants_vector_sizes: &mut Option<Vec<VectorSize>>,
    scheme: QuantScheme,
) {
    match scheme.store {
        QuantStore::Native => match scheme.value {
            // Type sizes are the same so just treat fp8/fp4x2 as i8
            QuantValue::Q8F
            | QuantValue::Q8S
            | QuantValue::E4M3
            | QuantValue::E5M2
            | QuantValue::E2M1 => {
                let vector_sizes = client
                    .io_optimized_vector_sizes(size_of::<i8>())
                    .collect::<Vec<_>>();

                match &quants_vector_sizes {
                    Some(sizes) => {
                        if sizes[0] < vector_sizes[0] {
                            *quants_vector_sizes = Some(vector_sizes);
                        }
                    }
                    None => {
                        *quants_vector_sizes = Some(vector_sizes);
                    }
                }
            }
            QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => {
                unreachable!("Can't store native sub-byte values")
            }
        },
        QuantStore::PackedU32(packed_dim) => {
            let mut vector_sizes = client
                .io_optimized_vector_sizes(size_of::<u32>())
                .collect::<Vec<_>>();

            for val in vector_sizes.iter_mut() {
                *val *= scheme.num_quants();
            }

            let min = *vector_sizes.last().unwrap();

            // We need to put back values that are not multiple of num_quants, but may be good
            // vectorization factor for other handles in a fused trace.
            for val in client.io_optimized_vector_sizes(size_of::<u32>()) {
                if val < min {
                    vector_sizes.push(val);
                }
            }

            if packed_dim != 0 {
                // A moved packed axis uses scalar gathers and unpacks one storage word at a time.
                // Keep output vectors at most as wide as the unpacked word to preserve the
                // dynamic vector type used by the dequantization kernel.
                vector_sizes.retain(|size| *size <= scheme.num_quants());
            }

            match &quants_vector_sizes {
                Some(sizes) => {
                    if sizes[0] < vector_sizes[0] {
                        let mut min = *vector_sizes.last().unwrap();

                        while min > 1 {
                            min /= 2;
                            vector_sizes.push(min);
                        }
                        *quants_vector_sizes = Some(vector_sizes);
                    }
                }
                None => {
                    *quants_vector_sizes = Some(vector_sizes);
                }
            }
        }
        QuantStore::PackedNative(_) => {
            panic!("Not yet supported")
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::codegen::ir::{FuseArg, FuseType, LayoutInfo};
    use burn_std::{Shape, Strides};

    fn concrete(shape: &[usize], strides: &[usize]) -> ReferenceSelection {
        ReferenceSelection::Concrete {
            layout: FuseArg::Output(0, FuseType::F32, LayoutInfo::IsRef),
            shape: Shape::from(shape.to_vec()),
            strides: Strides::new(strides),
        }
    }

    #[test]
    fn a_contiguous_reference_asks_for_nothing() {
        // The whole point: a block that iterates in logical dimension order has to
        // keep behaving exactly as it did before a block could choose its layout.
        let reference = concrete(&[2, 48, 16, 16], &[48 * 16 * 16, 16 * 16, 16, 1]);

        assert_eq!(permuted_innermost(&reference), None);
        assert_eq!(axis_vote(None, 4, None), AxisVote::Default);
    }

    #[test]
    fn a_virtual_reference_asks_for_nothing() {
        let reference = ReferenceSelection::Reshaped { reshape_pos: 0 };
        assert_eq!(permuted_innermost(&reference), None);

        let reference = ReferenceSelection::Searching;
        assert_eq!(permuted_innermost(&reference), None);
    }

    #[test]
    fn a_reference_that_is_not_dense_asks_for_nothing() {
        // A padded or sliced reference has no dimension order to impose.
        assert_eq!(permuted_innermost(&concrete(&[4, 8], &[16, 1])), None);
    }

    #[test]
    fn an_nhwc_reference_asks_for_its_channel_dimension() {
        let reference = concrete(&[2, 48, 16, 16], &[16 * 16 * 48, 1, 16 * 48, 48]);

        assert_eq!(permuted_innermost(&reference), Some(1));
        assert_eq!(axis_vote(Some(1), 4, None), AxisVote::Along(1));
    }

    #[test]
    fn a_permuted_reference_skips_its_degenerate_dimensions() {
        // The dimension order ends at a size-one dimension whose stride is zero.
        // Advancing a line along it would read the same element `width` times.
        let reference = concrete(&[2, 48, 16, 1], &[768, 1, 48, 0]);

        assert_eq!(permuted_innermost(&reference), Some(1));
    }

    #[test]
    fn an_output_the_block_cannot_write_in_lines_is_refused() {
        // A reshaped output on the recompute path is allocated contiguous while its
        // block iterates NHWC. `vectorization_output` is handed a shape and never
        // sees a stride, so nothing downstream would catch this.
        let contiguous = [48 * 16 * 16, 16 * 16, 16, 1];

        assert_eq!(
            axis_vote(Some(1), 4, Some(&contiguous)),
            AxisVote::Never,
            "stride along the block's innermost dimension is not one",
        );

        let nhwc = [16 * 16 * 48, 1, 16 * 48, 48];
        assert_eq!(axis_vote(Some(1), 4, Some(&nhwc)), AxisVote::Along(1));
    }

    #[test]
    fn an_output_of_a_lower_rank_is_refused_rather_than_indexed_past_its_end() {
        // A reshape can drop dimensions. Reading `shape[axis]` at the reference's
        // innermost index would panic; the default axis could never do that, since
        // it came from the tensor's own rank.
        assert_eq!(axis_vote(Some(3), 2, None), AxisVote::Never);
        assert_eq!(axis_vote(Some(2), 2, None), AxisVote::Never);
        assert_eq!(axis_vote(Some(1), 2, None), AxisVote::Along(1));
    }

    #[test]
    fn an_input_is_trusted_to_refuse_an_axis_it_cannot_use() {
        // Inputs are voted without strides on purpose: `vectorization_input` makes
        // the same check, and unlike this one it tells a broadcast dimension from a
        // dimension it merely cannot line up.
        assert_eq!(axis_vote(Some(1), 4, None), AxisVote::Along(1));
    }

    #[test]
    fn blocks_that_disagree_cancel_out() {
        // Modelled on the merge in `vectorization_axis`: any two votes that are not
        // the same answer leave the tensor unvectorized, including a permuted block
        // meeting one that only wants the default.
        let merge = |a: AxisVote, b: AxisVote| if a == b { a } else { AxisVote::Never };

        assert_eq!(
            merge(AxisVote::Along(1), AxisVote::Along(3)),
            AxisVote::Never
        );
        assert_eq!(
            merge(AxisVote::Along(1), AxisVote::Default),
            AxisVote::Never
        );
        assert_eq!(
            merge(AxisVote::Default, AxisVote::Default),
            AxisVote::Default
        );
        assert_eq!(
            merge(AxisVote::Along(1), AxisVote::Along(1)),
            AxisVote::Along(1)
        );
    }
}
