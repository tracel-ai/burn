#[cfg(feature = "autotune")]
use super::tune::fused_reduce_autotune;
#[cfg(feature = "autotune")]
use crate::tune::TunedTrace;
use crate::{
    CubeFusionHandle, FallbackOperation,
    base::strides_dyn_rank,
    engine::{
        codegen::ir::{FuseArg, FuseBlockConfig, FuseType, GlobalArgsLaunch, RefLayout},
        launch::{
            FuseTraceLauncher,
            runner::{TraceRunner, Vectorization},
        },
        tile::{TraceReadArgLaunch, TraceReadTileOperand},
        trace::{FuseTrace, TraceError, TuneOutput},
    },
    optim::{FusedOperation, elemwise::ElemwiseRunner},
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_fusion::stream::Context;
use burn_ir::ReduceDimOpIr;
use cubecl::{
    bytes::Bytes,
    client::Client,
    prelude::TensorBinding,
    tile::{Geometry, launch::Decline},
};
use cubek::microkernel::reduce::ReduceOperation;
use cubek::reduce::kernels::{
    ReduceCandidate, ReduceElements, ReduceKey, ReduceTensors, ReduceWorkspace,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub struct ReduceOptimization {
    pub(crate) info: Arc<ReduceOptimizationInfo>,
}

pub(crate) struct ReduceOptimizationInfo {
    pub(crate) trace: FuseTrace,
    /// The trace the fused reduce runs: `trace`'s read block, without its write block.
    trace_read: FuseTrace,
    trace_read_fallback: FuseTrace,
    trace_write_fallback: FuseTrace,
    pub(crate) client: Client,
    pub(crate) device: cubecl::Device,
    pub(crate) len: usize,
    pub(crate) len_read: usize,
    pub(crate) reduce: FusedReduce,
    settings: ReduceSettings,
}

impl ReduceOptimizationInfo {
    pub fn from_state(device: &cubecl::Device, state: ReduceOptimizationState) -> Self {
        let client = device.client();

        Self {
            trace: state.trace,
            trace_read: state.trace_read,
            trace_read_fallback: state.trace_read_fallback,
            trace_write_fallback: state.trace_write_fallback,
            client,
            device: device.clone(),
            len: state.len,
            len_read: state.len_read,
            reduce: state.reduce,
            settings: state.settings,
        }
    }
    pub fn to_state(&self) -> ReduceOptimizationState {
        ReduceOptimizationState {
            trace: self.trace.clone(),
            trace_read: self.trace_read.clone(),
            trace_read_fallback: self.trace_read_fallback.clone(),
            trace_write_fallback: self.trace_write_fallback.clone(),
            len: self.len,
            len_read: self.len_read,
            reduce: self.reduce.clone(),
            settings: self.settings,
        }
    }
}

#[derive(Serialize, Deserialize, Copy, Clone)]
pub enum ReduceSettings {
    Always,
    /// We only activate fuse-on-write when the reduction isn't on the last dimension, otherwise
    /// vectorization is impossible. Only `VectorizationMode::Perpendicular` supports vectorization.
    ///
    /// We could still fuse some output operations, but it would probably lead to worse performance.
    OnlyParallel,
    Never,
}

pub(crate) struct ReduceOptimizationTuneArg {
    pub(crate) info: Arc<ReduceOptimizationInfo>,
    pub(crate) fallback: Arc<Box<dyn FallbackOperation>>,
}

impl Clone for ReduceOptimizationTuneArg {
    fn clone(&self) -> Self {
        Self {
            info: self.info.clone(),
            fallback: self.fallback.clone(),
        }
    }
}

#[cfg(feature = "autotune")]
impl TunedTrace for ReduceOptimizationTuneArg {
    fn trace(&self) -> &FuseTrace {
        &self.info.trace
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub enum ReduceInstruction {
    ArgMax,
    ArgMin,
    Mean,
    Prod,
    Sum,
    Max,
    Min,
    MaxAbs,
    Any,
    All,
}

pub trait ReduceFallbackFn: Send + Sync {
    fn run(&self, context: &mut Context<CubeFusionHandle>);
}

#[derive(Serialize, Deserialize)]
pub struct ReduceOptimizationState {
    pub(crate) trace: FuseTrace,
    pub(crate) trace_read: FuseTrace,
    pub(crate) trace_read_fallback: FuseTrace,
    pub(crate) trace_write_fallback: FuseTrace,
    pub(crate) reduce: FusedReduce,
    pub(crate) len: usize,
    pub(crate) len_read: usize,
    pub(crate) settings: ReduceSettings,
}

impl core::fmt::Debug for ReduceOptimizationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{{ len_read: {}, len_total: {} }}",
            self.len_read, self.len
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusedReduce {
    pub(crate) input: FuseArg,
    pub(crate) output: FuseArg,
    pub(crate) acc: FuseType,
    pub(crate) axis: usize,
    pub(crate) op: ReduceDimOpIr,
    pub(crate) use_planes: bool,
    pub(crate) shared: bool,
    pub(crate) inst: ReduceInstruction,
}

/// A fused reduce on the routine cubek's reduce autotune elected for the unfused reduce: its input
/// read through the trace's read block, its output written to `out`.
#[derive(new)]
pub struct FusedReduceLaunch<'a> {
    reduce: &'a FusedReduce,
    device: &'a cubecl::Device,
    out: TensorBinding,
}

/// Why a fused reduce did not run fused; the fallback runs it instead.
#[derive(Debug)]
pub enum FusedReduceError {
    /// No routine is elected for the reduce: the autotune has not measured its key yet.
    NoRoutineElected,
    /// The elected routine declined the reduce, as its read block states it.
    Routine(Decline),
}

impl ReduceOptimizationTuneArg {
    /// Run the reduce: fused where the autotune measured that faster, the fallback otherwise.
    pub fn run(self, context: &mut Context<CubeFusionHandle>) {
        // An empty input leaves the fused launch either an axis it rejects or nothing to launch
        // over.
        let input = context.tensors.get(&self.info.reduce.op.input.id).unwrap();
        if input.shape.num_elements() == 0 {
            self.execute_fallback(context);
            return;
        }

        #[cfg(feature = "autotune")]
        fused_reduce_autotune(self, context);

        #[cfg(not(feature = "autotune"))]
        if self.execute_fused(context).is_err() {
            self.execute_fallback(context);
        }
    }

    /// The read block and the reduce in one launch, into a new output, then the write block on
    /// that output in a second one.
    pub fn execute_fused(
        &self,
        context: &mut Context<CubeFusionHandle>,
    ) -> Result<TuneOutput, TraceError<FusedReduceError>> {
        let out_desc = context.tensors.get(&self.info.reduce.op.out.id).unwrap();
        let (out_id, out_shape) = (out_desc.id, out_desc.shape.clone());
        let size = out_shape.num_elements() * dtype_to_storage_type(out_desc.dtype).size();
        let out = CubeFusionHandle {
            client: self.info.client.clone(),
            handle: self.info.client.empty(size),
            device: self.info.device.clone(),
            strides: strides_dyn_rank(&out_shape),
            tiles: None,
            dtype: out_desc.dtype,
            qparams: None,
        };

        let launch = FusedReduceLaunch::new(
            &self.info.reduce,
            &self.info.device,
            out.clone().binding(out_shape.clone()),
        );
        let launcher = FuseTraceLauncher::new(&self.info.trace_read, &launch);
        #[allow(unused_mut)] // It is used when `autotune-checks` is activated.
        let mut output_read = launcher.launch(&self.info.client, &self.info.device, context)?;

        context.handles.register_handle(out_id, out.clone());

        #[cfg(feature = "autotune-checks")]
        if let TuneOutput::Checked { handles } = &mut output_read {
            handles.insert(self.info.reduce.op.out.id, (out_shape, out));
        }

        let launcher = FuseTraceLauncher::new(&self.info.trace_write_fallback, &ElemwiseRunner);
        let output_write = launcher
            .launch(&self.info.client, &self.info.device, context)
            .unwrap();

        Ok(output_read.merge(output_write))
    }

    pub fn execute_fallback(&self, context: &mut Context<CubeFusionHandle>) -> TuneOutput {
        let launcher = FuseTraceLauncher::new(&self.info.trace_read_fallback, &ElemwiseRunner);

        #[allow(unused_mut)] // It is used when `autotune-checks` is activated.
        let mut output_read = launcher
            .launch(&self.info.client, &self.info.device, context)
            .unwrap();

        self.fallback.run(context);

        #[cfg(feature = "autotune-checks")]
        if let TuneOutput::Checked { handles } = &mut output_read {
            let out_desc = context.tensors.get(&self.info.reduce.op.out.id).unwrap();
            let handle_out = context
                .handles
                .get_handle(&out_desc.id, &burn_ir::TensorStatus::ReadOnly);

            handles.insert(
                self.info.reduce.op.out.id,
                (out_desc.shape.clone(), handle_out.clone()),
            );
        }

        let launcher = FuseTraceLauncher::new(&self.info.trace_write_fallback, &ElemwiseRunner);

        let output_write = launcher
            .launch(&self.info.client, &self.info.device, context)
            .unwrap();

        output_read.merge(output_write)
    }
}

#[allow(clippy::too_many_arguments)]
impl ReduceOptimization {
    pub fn new(
        trace: FuseTrace,
        trace_read: FuseTrace,
        trace_read_fallback: FuseTrace,
        trace_write_fallback: FuseTrace,
        client: Client,
        device: cubecl::Device,
        len: usize,
        len_read: usize,
        reduce: FusedReduce,
        settings: ReduceSettings,
    ) -> Self {
        let info = ReduceOptimizationInfo {
            trace,
            trace_read,
            trace_read_fallback,
            trace_write_fallback,
            client,
            device,
            len,
            len_read,
            reduce,
            settings,
        };

        Self {
            info: Arc::new(info),
        }
    }
    /// Execute the optimization.
    pub fn execute(
        &mut self,
        context: &mut Context<CubeFusionHandle>,
        fallback: impl FnOnce(usize) -> Box<dyn FallbackOperation>,
    ) {
        // The index of the fallback reduce is the number of ops fused as read.
        let fallback = fallback(self.info.len_read);
        ReduceOptimizationTuneArg {
            info: self.info.clone(),
            fallback: Arc::new(fallback),
        }
        .run(context);
    }

    pub fn num_output_buffers(&self) -> usize {
        self.info.trace_read_fallback.resources.outputs.len()
    }

    pub fn to_state(&self) -> ReduceOptimizationState {
        ReduceOptimizationState {
            trace: self.info.trace.clone(),
            trace_read: self.info.trace_read.clone(),
            trace_read_fallback: self.info.trace_read_fallback.clone(),
            trace_write_fallback: self.info.trace_write_fallback.clone(),
            reduce: self.info.reduce.clone(),
            len: self.info.len,
            len_read: self.info.len_read,
            settings: self.info.settings,
        }
    }

    pub fn from_state(device: &cubecl::Device, state: ReduceOptimizationState) -> Self {
        let client = device.client();

        let info = ReduceOptimizationInfo {
            trace: state.trace,
            trace_read: state.trace_read,
            trace_read_fallback: state.trace_read_fallback,
            trace_write_fallback: state.trace_write_fallback,
            reduce: state.reduce,
            len: state.len,
            len_read: state.len_read,
            client,
            device: device.clone(),
            settings: state.settings,
        };

        Self {
            info: Arc::new(info),
        }
    }

    /// Returns the number of output buffers added by fusion.
    pub fn num_ops_fused(&self) -> usize {
        self.info.len
    }
}

impl Vectorization for FusedReduceLaunch<'_> {}

impl TraceRunner for FusedReduceLaunch<'_> {
    type Error = FusedReduceError;

    fn run<'a>(
        &'a self,
        client: &'a Client,
        inputs: GlobalArgsLaunch,
        outputs: GlobalArgsLaunch,
        configs: &'a [FuseBlockConfig],
    ) -> Result<(), FusedReduceError> {
        let config_read = &configs[0];
        let geometry = block_geometry(&inputs, &outputs, config_read);
        let operation = reduce_operation(self.reduce.inst);
        let elements = ReduceElements {
            input: dtype_to_storage_type(self.reduce.op.input.dtype),
            accumulator: self.reduce.acc.into_elem(),
            out: dtype_to_storage_type(self.reduce.op.out.dtype),
        };
        let workspace = reduce_workspace(
            client,
            geometry.shape(),
            self.reduce.axis,
            operation,
            elements,
        );
        let tensors = ReduceTensors::unbound(
            client,
            geometry,
            config_read.width,
            self.out.clone(),
            self.reduce.axis,
            operation,
            elements,
        )
        .with_workspace(workspace);
        let candidate = ReduceCandidate::fastest(client, self.device, &ReduceKey::from(&tensors))
            .ok_or(FusedReduceError::NoRoutineElected)?;
        candidate
            .launch_unbound::<TraceReadTileOperand>(&tensors, |source| {
                TraceReadArgLaunch::of(
                    inputs,
                    outputs,
                    source,
                    self.reduce.input.clone(),
                    config_read.clone(),
                )
            })
            .map_err(FusedReduceError::Routine)
    }
}

/// The scratch cubek's reductions that split a row across cubes take for reducing `axis` of an
/// input of `shape` by `operation` in `elements`: cubek allocates none, so the fused launch does,
/// sized for whichever kernel the autotune elected, its arrival counters zero.
fn reduce_workspace(
    client: &Client,
    shape: &[usize],
    axis: usize,
    operation: ReduceOperation,
    elements: ReduceElements,
) -> ReduceWorkspace {
    let bytes = ReduceWorkspace::bytes_for(client, shape, axis, operation, elements, false);
    // A runtime may refuse an empty allocation: every buffer holds at least one counter.
    let at_least_one = |bytes: u64| (bytes as usize).max(size_of::<u32>());
    let counters = at_least_one(bytes.arrivals) / size_of::<u32>();
    ReduceWorkspace {
        partials: client.empty(at_least_one(bytes.partials)),
        partial_indices: client.empty(at_least_one(bytes.partial_indices)),
        arrivals: client.create(Bytes::from_elems(vec![0u32; counters])),
    }
}

/// The shape and strides of `config`'s reference layout, which its block reads or writes lines of.
fn block_geometry(
    inputs: &GlobalArgsLaunch,
    outputs: &GlobalArgsLaunch,
    config: &FuseBlockConfig,
) -> Geometry {
    // An output-concrete reference indexes the output arguments; shape and strides must both be
    // resolved against that list.
    let args = match &config.ref_layout {
        RefLayout::Concrete(FuseArg::Output(..)) => outputs,
        _ => inputs,
    };
    let shape = args.shape_ref(&config.ref_layout, config.rank);
    let strides = args.strides_ref(&config.ref_layout, config.rank);
    let dims: Vec<(usize, usize)> = shape.iter().copied().zip(strides.iter().copied()).collect();
    Geometry::new(&dims)
}

/// The operation `instruction` names, as the reduce microkernels name it.
pub(crate) fn reduce_operation(instruction: ReduceInstruction) -> ReduceOperation {
    match instruction {
        ReduceInstruction::ArgMax => ReduceOperation::ArgMax,
        ReduceInstruction::ArgMin => ReduceOperation::ArgMin,
        ReduceInstruction::Prod => ReduceOperation::Prod,
        ReduceInstruction::Mean => ReduceOperation::Mean,
        ReduceInstruction::Sum => ReduceOperation::Sum,
        ReduceInstruction::Max => ReduceOperation::Max,
        ReduceInstruction::Min => ReduceOperation::Min,
        ReduceInstruction::MaxAbs => ReduceOperation::MaxAbs,
        ReduceInstruction::Any => ReduceOperation::Any,
        ReduceInstruction::All => ReduceOperation::All,
    }
}

/// Name of the reduce fusion optimization.
pub const NAME: &str = "Reduce";

impl ReduceOptimizationInfo {
    /// The highest relative shape id across this reduce's trace and both of its fallbacks.
    pub(crate) fn max_relative_shape_id(&self) -> Option<usize> {
        [
            self.trace.max_relative_shape_id(),
            self.trace_read.max_relative_shape_id(),
            self.trace_read_fallback.max_relative_shape_id(),
            self.trace_write_fallback.max_relative_shape_id(),
        ]
        .into_iter()
        .flatten()
        .max()
    }
}

impl FusedOperation for ReduceOptimization {
    fn max_relative_shape_id(&self) -> Option<usize> {
        self.info.max_relative_shape_id()
    }

    const NAME: &'static str = self::NAME;
    type State = ReduceOptimizationState;

    fn num_ops_fused(&self) -> usize {
        Self::num_ops_fused(self)
    }

    fn run(
        &mut self,
        context: &mut Context<CubeFusionHandle>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation>,
    ) {
        Self::execute(self, context, |index| fallback(index))
    }

    fn to_state(&self) -> Self::State {
        Self::to_state(self)
    }

    fn from_state(device: &cubecl::Device, state: Self::State) -> Self {
        Self::from_state(device, state)
    }
}
