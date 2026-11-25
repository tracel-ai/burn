//! This backend's autotunes, read by the fused operations that launch the routine an unfused one
//! elected ([`CubeRoutineElector`]).

use burn_cubecl_fusion::RoutineElector;
use cubecl::{Device, client::Client};
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
}
