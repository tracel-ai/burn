use burn_fusion::stream::Context;
use burn_ir::{TensorId, TensorIr};
use burn_tensor::DType;
use cubecl::{client::ComputeClient, ir::Elem, CubeElement, Runtime};

use crate::{
    elem_dtype, is_contiguous,
    on_write::ir::{Arg, ElemwiseOp, LayoutInfo},
    strides_dyn_rank, CubeFusionHandle,
};

use super::{
    super::ir::ElemwisePrecision, HandleOutput, LaunchPlan, Reference, RegisteredTensors,
    TensorView,
};

/// Create or reuse handles for the outputs.
///
/// It is also responsible to select the reference tensor.
pub struct OutputPlanner<'a, R: Runtime> {
    inputs: &'a RegisteredTensors,
    views: &'a Vec<TensorView>,
    outputs_sorted: Vec<OutputSorted<'a>>,
    handles: Vec<Option<HandleOutput<R>>>,
    globals: Vec<Option<TensorIr>>,
}

struct OutputSorted<'a> {
    pos_original: usize,
    precision: ElemwisePrecision,
    tensor_relative: &'a TensorIr,
}

enum OutputKind {
    Normal,
    Inplace {
        /// The position in the potential inplace vector
        input_pos: usize,
    },
    Transform(TensorView),
}

impl<'a, R: Runtime> OutputPlanner<'a, R> {
    pub fn new(
        inputs: &'a RegisteredTensors,
        outputs: &'a RegisteredTensors,
        views: &'a Vec<TensorView>,
    ) -> Self {
        let mut outputs_sorted: Vec<_> = outputs
            .iter()
            .enumerate()
            .map(|(pos, (tensor, precision))| OutputSorted {
                pos_original: pos,
                precision: *precision,
                tensor_relative: tensor,
            })
            .collect();

        outputs_sorted.sort_by(|a, b| {
            let a_val: usize = a.tensor_relative.shape.iter().sum();
            let b_val: usize = b.tensor_relative.shape.iter().sum();

            b_val.cmp(&a_val)
        });

        let mut handles = Vec::with_capacity(outputs.len());
        let mut globals = Vec::with_capacity(outputs.len());

        for _ in 0..outputs.len() {
            handles.push(None);
            globals.push(None);
        }

        Self {
            inputs,
            outputs_sorted,
            views,
            handles,
            globals,
        }
    }

    pub fn run<BT: CubeElement>(
        mut self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
    ) {
        // So that we can borrow self during the iteration.
        let mut outputs = Vec::new();
        core::mem::swap(&mut outputs, &mut self.outputs_sorted);

        for output in outputs.into_iter() {
            let tensor_global = context
                .tensors
                .get(&output.tensor_relative.id)
                .unwrap()
                .clone();
            let strides = strides_dyn_rank(&tensor_global.shape);

            match self.output_kind(plan, &tensor_global, &output, &strides) {
                OutputKind::Inplace { input_pos } => {
                    self.inplace_output(context, plan, output, tensor_global, input_pos);
                }
                OutputKind::Normal => {
                    self.normal_output::<BT>(
                        client,
                        device,
                        context,
                        plan,
                        output,
                        tensor_global,
                        strides,
                    );
                }
                OutputKind::Transform(TensorView::Reshape { original, .. }) => {
                    self.reshaped_output::<BT>(
                        client,
                        device,
                        context,
                        plan,
                        output,
                        tensor_global,
                        strides,
                        original,
                    );
                }
                OutputKind::Transform(TensorView::SwapDims { original, dims, .. }) => {
                    self.swapped_dims_output::<BT>(
                        client,
                        device,
                        context,
                        plan,
                        output,
                        tensor_global,
                        original,
                        dims,
                    );
                }
            }
        }

        for (handle, global) in self.handles.into_iter().zip(self.globals.into_iter()) {
            plan.handle_outputs.push(handle.unwrap());
            plan.global_outputs.push(global.unwrap());
        }

        Self::add_layout_info_inputs(plan);
    }

    fn add_layout_info_inputs(plan: &mut LaunchPlan<'_, R>) {
        for hi in plan.handle_inputs.iter() {
            if let Some(reference) = plan.reference.as_ref() {
                if reference.strides == hi.handle.strides && reference.shape == hi.global_shape {
                    if let Some(ops) = plan.reads.get_mut(&hi.relative_id) {
                        for op in ops.iter_mut() {
                            if let ElemwiseOp::Assign(op) = op {
                                op.input.add_layout_info(LayoutInfo::SameAsRef);
                            }
                        }
                    }
                }
            }
        }
    }

    fn output_kind(
        &self,
        plan: &mut LaunchPlan<'a, R>,
        tensor_global: &TensorIr,
        output: &OutputSorted,
        strides: &[usize],
    ) -> OutputKind {
        if let Some(transform) = self.views.iter().find(|v| match v {
            TensorView::Reshape { reshaped, .. } => reshaped == &output.tensor_relative.id,
            TensorView::SwapDims { swapped, .. } => swapped == &output.tensor_relative.id,
        }) {
            return OutputKind::Transform(transform.clone());
        }

        plan.potential_inplaces
            .iter()
            .enumerate()
            .find(|(_pos, pi)| {
                pi.tensor_relative.dtype == tensor_global.dtype
                    && pi.tensor_relative.shape == output.tensor_relative.shape
                    && pi.strides == strides
            })
            .map(|(pos, _)| OutputKind::Inplace { input_pos: pos })
            .unwrap_or(OutputKind::Normal)
    }

    fn inplace_output(
        &mut self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorIr,
        input_index: usize,
    ) {
        let potential_inplace = plan.potential_inplaces.remove(input_index);
        let handle_input = plan.handle_inputs.get(potential_inplace.input_pos).unwrap();

        if plan.reference.is_none() {
            let index_input = self
                .inputs
                .get_index(potential_inplace.tensor_relative.id)
                .unwrap();

            plan.reference = Some(Reference {
                layout: Arg::Input(index_input, output.precision, LayoutInfo::IsRef),
                shape: tensor_global.shape.clone(),
                strides: handle_input.handle.strides.clone(),
            });

            if let Some(ops) = plan.reads.get_mut(&handle_input.relative_id) {
                for op in ops.iter_mut() {
                    if let ElemwiseOp::Assign(op) = op {
                        op.input.add_layout_info(LayoutInfo::IsRef);
                    };
                }
            }

            if let Some(ElemwiseOp::Assign(op)) = plan.writes.get_mut(&output.tensor_relative.id) {
                op.out.add_layout_info(LayoutInfo::IsRef);
            };
        }

        context
            .handles
            .register_handle(tensor_global.id, handle_input.handle.clone());

        self.handles[output.pos_original] = Some(HandleOutput::Alias {
            input_pos: potential_inplace.input_pos,
            precision: output.precision,
        });
        self.globals[output.pos_original] = Some(tensor_global);
    }

    #[allow(clippy::too_many_arguments)]
    fn normal_output<BT: CubeElement>(
        &mut self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorIr,
        strides: Vec<usize>,
    ) {
        if plan.reference.is_none() {
            plan.reference = Some(Reference {
                layout: Arg::Output(
                    output.pos_original as u32,
                    output.precision,
                    LayoutInfo::IsRef,
                ),
                shape: tensor_global.shape.clone(),
                strides: strides.clone(),
            });

            if let ElemwiseOp::Assign(op) = plan.writes.get_mut(&output.tensor_relative.id).unwrap()
            {
                op.out.add_layout_info(LayoutInfo::IsRef);
            };
        } else if let Some(reference) = plan.reference.as_ref() {
            if reference.strides == strides && reference.shape == tensor_global.shape {
                if let ElemwiseOp::Assign(op) =
                    plan.writes.get_mut(&output.tensor_relative.id).unwrap()
                {
                    op.out.add_layout_info(LayoutInfo::SameAsRef);
                };
            }
        }

        // We encode bool tensors as `B`.
        let dtype = match tensor_global.dtype {
            DType::Bool => elem_dtype::<BT>(),
            _ => tensor_global.dtype,
        };
        let size = tensor_global.shape.iter().product::<usize>() * Elem::from(dtype).size();

        let handle = CubeFusionHandle {
            client: client.clone(),
            handle: client.empty(size),
            device: device.clone(),
            strides,
            dtype,
        };

        plan.rank = usize::max(tensor_global.shape.len(), plan.rank);
        context
            .handles
            .register_handle(tensor_global.id, handle.clone());

        self.handles[output.pos_original] = Some(HandleOutput::Owned {
            precision: output.precision,
            handle,
            global_shape: tensor_global.shape.clone(),
            global_id: tensor_global.id,
            vectorization: 1,
        });
        self.globals[output.pos_original] = Some(tensor_global);
    }

    #[allow(clippy::too_many_arguments)]
    fn reshaped_output<BT: CubeElement>(
        &mut self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorIr,
        strides: Vec<usize>,
        original: TensorId,
    ) {
        let (pos_input, original_handle) = plan
            .handle_inputs
            .iter()
            .enumerate()
            .find(|(_i, handle)| handle.relative_id == original)
            .unwrap();

        // We encode bool tensors as `B`.
        let dtype = match tensor_global.dtype {
            DType::Bool => elem_dtype::<BT>(),
            _ => tensor_global.dtype,
        };

        if is_contiguous(
            &original_handle.global_shape,
            &original_handle.handle.strides,
        ) {
            plan.writes.remove(&output.tensor_relative.id);

            let handle = CubeFusionHandle {
                client: client.clone(),
                handle: original_handle.handle.handle.clone(),
                device: device.clone(),
                strides,
                dtype,
            };
            context
                .handles
                .register_handle(tensor_global.id, handle.clone());

            // IT will never be access, just a way to keep the original position working.
            self.handles[output.pos_original] = Some(HandleOutput::Alias {
                input_pos: pos_input,
                precision: output.precision,
            });
            self.globals[output.pos_original] = Some(tensor_global);
        } else {
            self.normal_output::<BT>(
                client,
                device,
                context,
                plan,
                output,
                tensor_global,
                strides,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn swapped_dims_output<BT: CubeElement>(
        &mut self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        plan: &mut LaunchPlan<'a, R>,
        output: OutputSorted,
        tensor_global: TensorIr,
        original: TensorId,
        dims: (u32, u32),
    ) {
        let (pos_input, original_handle) = plan
            .handle_inputs
            .iter()
            .enumerate()
            .find(|(_i, handle)| handle.relative_id == original)
            .unwrap();

        // We encode bool tensors as `B`.
        let dtype = match tensor_global.dtype {
            DType::Bool => elem_dtype::<BT>(),
            _ => tensor_global.dtype,
        };

        // TODO: Check if we can also remove the read, if we have a dead partial graph.
        plan.writes.remove(&output.tensor_relative.id);

        let strides = original_handle.handle.strides.clone();
        let mut handle = CubeFusionHandle {
            client: client.clone(),
            handle: original_handle.handle.handle.clone(),
            device: device.clone(),
            strides,
            dtype,
        };
        handle.strides.swap(dims.0 as usize, dims.1 as usize);

        context
            .handles
            .register_handle(tensor_global.id, handle.clone());

        // IT will never be access, just a way to keep the original position working.
        self.handles[output.pos_original] = Some(HandleOutput::Alias {
            input_pos: pos_input,
            precision: output.precision,
        });
        self.globals[output.pos_original] = Some(tensor_global);
    }
}
