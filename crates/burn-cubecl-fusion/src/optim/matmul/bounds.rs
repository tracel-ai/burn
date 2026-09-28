use super::{optimization::MatmulOptimizationTuneArg, tune::FusedMatmulAutotuneKey};
use crate::{engine::trace::TuneOutput, tune::FusionTuneInputs};
use burn_backend::cubecl::{autotune::with_roofline_bounds, dtype_to_storage_type};
use cubecl::{
    std::throughput::roofline_bounds,
    tune::{TunableSet, Work},
};
use cubek::matmul::definition::{MatmulCost, MatmulGlobalElems};

type Inputs = FusionTuneInputs<MatmulOptimizationTuneArg>;

/// Registers the roofline of the product together with its tail: the product's arithmetic, and
/// the traffic of the whole trace.
pub(crate) fn with_fused_matmul_bounds(
    set: TunableSet<FusedMatmulAutotuneKey, Inputs, TuneOutput>,
) -> TunableSet<FusedMatmulAutotuneKey, Inputs, TuneOutput> {
    with_roofline_bounds(set, |_key, inputs, thresholds| {
        let info = &inputs.optimization().info;
        let tensors = inputs.tensors();
        let tensor = |id| {
            tensors
                .get(id)
                .expect("matmul operand registered in the context")
        };
        let matmul = &info.matmul;
        let (lhs, out) = (tensor(&matmul.op.lhs.id), tensor(&matmul.op.out.id));
        let rank = out.shape.len();
        // The operands at the precision the launch computes them in: a packed operand's dtype
        // names its storage word, which no matrix instruction takes.
        let cost = MatmulCost {
            batches: out.shape[..rank - 2].iter().product(),
            m: out.shape[rank - 2],
            k: lhs.shape[rank - 1],
            n: out.shape[rank - 1],
            elems: MatmulGlobalElems {
                lhs: matmul.lhs.precision().into_storage_type(),
                rhs: matmul.rhs.precision().into_storage_type(),
                out: dtype_to_storage_type(out.dtype),
            },
        };
        let work = Work {
            compute_ops: cost.compute_ops(),
            bytes: info.trace.traffic(tensors),
        };

        roofline_bounds(
            &info.client,
            cost.compute_key(&info.client),
            work,
            thresholds,
        )
    })
}
