use super::{
    TileFusedMatmulError,
    sink::{TraceSink, TraceSinkArgLaunch},
};
use crate::{
    CubeFusionHandle,
    engine::{
        codegen::ir::{FuseArg, FuseBlockConfig, GlobalArgsLaunch, RefLayout},
        launch::{
            runner::{TraceRunner, Vectorization, VectorizationAxis, VectorizationHandle},
            vectorization::{Vect, vectorization_default},
        },
    },
};
use burn_fusion::stream::Context;
use burn_ir::{TensorId, TensorIr};
use core::cell::Cell;
use cubecl::{client::Client, ir::VectorSize, tile::launch::SinkLaunch};
use std::collections::BTreeMap;

/// A routine's product, prepared for a sink, run by the trace launcher with the trace's block as
/// its output.
///
/// The routine binds its own operands before the launcher runs, so the launch rides here, taken
/// once: [`TraceRunner::run`] takes `&self`.
pub(super) struct Carrier<L> {
    launch: Cell<Option<L>>,
    product: FuseArg,
    /// The width the routine stores its lines in, which the block must store in too.
    width: usize,
    /// The elements the routine stores, which the block must cover exactly.
    elements: usize,
}

impl<L: SinkLaunch> Carrier<L> {
    pub(super) fn new(launch: L, product: FuseArg) -> Self {
        Self {
            width: launch.width(),
            elements: launch.elements(),
            launch: Cell::new(Some(launch)),
            product,
        }
    }
}

/// Burn's planning, under the routine's store width: the block is planned around a kernel whose
/// width is already settled.
impl<L> Vectorization for Carrier<L> {
    fn vectorization<'a>(
        &self,
        _context: &Context<CubeFusionHandle>,
        vectorizations: &mut BTreeMap<TensorId, Vect>,
        inputs: impl Iterator<Item = VectorizationHandle<'a>>,
        outputs: impl Iterator<Item = &'a TensorIr>,
        reshaped: impl Iterator<Item = (&'a TensorIr, &'a TensorIr, bool)>,
        swapped: impl Iterator<Item = (&'a TensorIr, &'a TensorIr, bool, &'a (usize, usize))>,
        vector_sizes: &[VectorSize],
        max: VectorSize,
        axis: VectorizationAxis,
    ) {
        let fitting: Vec<VectorSize> = vector_sizes
            .iter()
            .copied()
            .filter(|&size| size <= self.width)
            .collect();
        vectorization_default(
            vectorizations,
            inputs,
            outputs,
            reshaped,
            swapped,
            &fitting,
            &Default::default(),
            max.min(self.width),
            &axis,
        )
    }
}

impl<L: SinkLaunch> TraceRunner for Carrier<L> {
    type Error = TileFusedMatmulError;

    fn run<'a>(
        &'a self,
        _client: &'a Client,
        inputs: GlobalArgsLaunch,
        outputs: GlobalArgsLaunch,
        configs: &'a [FuseBlockConfig],
    ) -> Result<(), TileFusedMatmulError> {
        let launch = self.launch.take().ok_or(TileFusedMatmulError::RanTwice)?;
        let [config] = configs else {
            return Err(TileFusedMatmulError::NotOneBlock {
                blocks: configs.len(),
            });
        };
        // The routine's write position counts lines of its width, the block's lines of its own:
        // any other block width would read the tail's inputs at another element than the one
        // being written.
        if config.width != self.width {
            return Err(TileFusedMatmulError::BlockWidth {
                block: config.width,
                store: self.width,
            });
        }
        // The routine writes its own count of elements: a block covering another count would be
        // left partly unwritten, or run past the product.
        let reference = match &config.ref_layout {
            RefLayout::Concrete(FuseArg::Output(..)) => {
                outputs.shape_ref(&config.ref_layout, config.rank)
            }
            _ => inputs.shape_ref(&config.ref_layout, config.rank),
        };
        let covered: usize = reference.iter().product();
        if covered != self.elements {
            return Err(TileFusedMatmulError::BlockCoverage {
                block: covered,
                store: self.elements,
            });
        }
        let (product, config) = (self.product.clone(), config.clone());
        launch.launch::<TraceSink>(|sink| {
            TraceSinkArgLaunch::of(inputs, outputs, sink, product, config)
        });
        Ok(())
    }
}
