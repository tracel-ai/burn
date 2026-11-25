//! This backend's autotunes, read by the fused operations that launch the routine an unfused one
//! elected ([`CubeRoutineElector`]).

use crate::kernel::matmul::{MatmulAutotuneGroups, fastest_routine};
use burn_cubecl_fusion::RoutineElector;
use cubecl::{Device, client::Client};
use cubek::matmul::kernels::MatmulRoutineSettings;
use cubek::matmul::kernels::plan::MatmulCall;
use cubek::matmul::kernels::select::MatmulAutotune;
use cubek::reduce::kernels::{ReduceCandidate, ReduceKey};

/// Reads the winners of this backend's autotunes for the fusion crate, which cannot reach the
/// tuners itself: they are this crate's, and this crate depends on it.
pub(crate) struct CubeRoutineElector;

/// The elector the fused operations read, installed before any of them is built or restored.
pub(crate) static ELECTOR: CubeRoutineElector = CubeRoutineElector;

impl RoutineElector for CubeRoutineElector {
    #[cfg_attr(not(feature = "autotune"), allow(unused_variables))]
    fn elect_reduce(
        &self,
        client: &Client,
        device: &Device,
        key: &ReduceKey,
    ) -> Option<ReduceCandidate> {
        #[cfg(feature = "autotune")]
        return crate::kernel::reduce::fastest_tuned(client, device, key);

        #[cfg(not(feature = "autotune"))]
        None
    }

    /// The matmul autotune's tuner is built under every feature set: the unfused product's plan
    /// races on it whenever its strategy measures.
    fn elect_matmul(
        &self,
        autotune: MatmulAutotune,
        call: &MatmulCall,
    ) -> Option<MatmulRoutineSettings> {
        fastest_routine(autotune, call, MatmulAutotuneGroups::default())
    }
}
