use crate::{
    CubeFusionHandle, FallbackOperation,
    engine::{
        codegen::ir::FuseArg,
        launch::FuseTraceLauncher,
        trace::{FuseTrace, TuneOutput},
    },
    optim::{FusedOperation, elemwise::ElemwiseRunner, matmul::args::MatmulArg},
    tune::TunedTrace,
};
use burn_fusion::stream::Context;
use burn_ir::BinaryOpIr;
use cubecl::client::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Fuse matmul operation followed by elemwise operations into a single kernel.
pub struct MatmulOptimization {
    pub(crate) info: Arc<MatmulOptimizationInfo>,
}

pub struct MatmulOptimizationTuneArg {
    pub(crate) info: Arc<MatmulOptimizationInfo>,
    pub(crate) fallback: Box<dyn FallbackOperation>,
}

impl TunedTrace for MatmulOptimizationTuneArg {
    fn trace(&self) -> &FuseTrace {
        &self.info.trace
    }
}

pub(crate) struct MatmulOptimizationInfo {
    pub(crate) trace: FuseTrace,
    trace_fallback: FuseTrace,
    pub(crate) client: Client,
    pub(crate) device: cubecl::Device,
    pub(crate) len: usize,
    pub(crate) matmul: FusedMatmul,
}

#[derive(Serialize, Deserialize, Debug)]
/// State for the [matrix optimization](MatmulOptimizationState).
pub struct MatmulOptimizationState {
    trace: FuseTrace,
    trace_fallback: FuseTrace,
    matmul: FusedMatmul,
    len: usize,
}

impl MatmulOptimizationInfo {
    /// Returns the number of output buffers added by fusion.
    #[cfg(feature = "autotune")]
    pub fn num_output_buffers(&self) -> usize {
        self.trace_fallback.resources.outputs.len()
    }

    /// Number of operations fused.
    pub fn num_ops_fused(&self) -> usize {
        self.len
    }
}

impl MatmulOptimizationTuneArg {
    pub fn execute_fallback(&self, context: &mut Context<CubeFusionHandle>) -> TuneOutput {
        self.fallback.run(context);

        #[cfg(feature = "autotune-checks")]
        let mut output = TuneOutput::Checked {
            handles: Default::default(),
        };
        #[cfg(not(feature = "autotune-checks"))]
        let output = TuneOutput::UnChecked;

        #[cfg(feature = "autotune-checks")]
        if let TuneOutput::Checked { handles } = &mut output {
            let out_desc = context.tensors.get(&self.info.matmul.op.out.id).unwrap();
            let handle_out = context
                .handles
                .get_handle(&out_desc.id, &burn_ir::TensorStatus::ReadOnly);

            handles.insert(
                self.info.matmul.op.out.id,
                (out_desc.shape.clone(), handle_out.clone()),
            );
        }

        let launcher = FuseTraceLauncher::new(&self.info.trace_fallback, &ElemwiseRunner);
        let output_write = launcher
            .launch(&self.info.client, &self.info.device, context)
            .unwrap();

        output.merge(output_write)
    }
}

impl MatmulOptimization {
    pub fn new(
        trace: FuseTrace,
        trace_fallback: FuseTrace,
        client: Client,
        device: cubecl::Device,
        len: usize,
        matmul: FusedMatmul,
    ) -> Self {
        let info = MatmulOptimizationInfo {
            trace,
            trace_fallback,
            client,
            device,
            len,
            matmul,
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
        // The index of the fallback matmul is always 0.
        let fallback = fallback(0);
        let arg = MatmulOptimizationTuneArg {
            info: self.info.clone(),
            fallback,
        };

        // A storage-tiled operand is read through its binding by the eager matmul, which stages
        // to its tiles; the fused product is not verified on them.
        let op = &arg.info.matmul.op;
        let stored = [op.lhs.id, op.rhs.id].iter().any(|id| {
            context
                .handles
                .get_handle_ref(id)
                .is_some_and(|handle| handle.tiles.is_some())
        });
        if stored {
            arg.execute_fallback(context);
            return;
        }

        // The fused product runs the routine the unfused one measured fastest, and races the
        // fallback; without autotune nothing is measured, so the fallback runs.
        #[cfg(feature = "autotune")]
        super::tile::execute(arg, context);

        #[cfg(not(feature = "autotune"))]
        arg.execute_fallback(context);
    }

    /// Number of operations fused.
    pub fn num_ops_fused(&self) -> usize {
        self.info.num_ops_fused()
    }

    /// Create an optimization from its [state](MatmulOptimizationState).
    pub fn from_state(device: &cubecl::Device, state: MatmulOptimizationState) -> Self {
        let info = MatmulOptimizationInfo {
            trace: state.trace,
            trace_fallback: state.trace_fallback,
            len: state.len,
            client: device.client(),
            device: device.clone(),
            matmul: state.matmul.clone(),
        };

        Self {
            info: Arc::new(info),
        }
    }

    /// Convert the optimization to its [state](MatmulOptimizationState).
    pub fn to_state(&self) -> MatmulOptimizationState {
        MatmulOptimizationState {
            trace: self.info.trace.clone(),
            trace_fallback: self.info.trace_fallback.clone(),
            matmul: self.info.matmul.clone(),
            len: self.info.len,
        }
    }
}

#[derive(new, Clone, Serialize, Deserialize, Debug)]
pub struct FusedMatmul {
    pub(crate) lhs: MatmulArg,
    pub(crate) rhs: MatmulArg,
    pub(crate) out: FuseArg,
    pub(crate) op: BinaryOpIr,
}

/// Name of the matmul fusion optimization.
pub const NAME: &str = "Matmul";

impl FusedOperation for MatmulOptimization {
    fn max_relative_shape_id(&self) -> Option<usize> {
        [
            self.info.trace.max_relative_shape_id(),
            self.info.trace_fallback.max_relative_shape_id(),
        ]
        .into_iter()
        .flatten()
        .max()
    }

    const NAME: &'static str = self::NAME;
    type State = MatmulOptimizationState;

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
