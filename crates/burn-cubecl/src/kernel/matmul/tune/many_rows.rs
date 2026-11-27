use super::base::{MatmulAutotuneGroups, MatmulTunables, RoutineIdentity, candidate_list};
use super::candidate::MatmulTuneCandidate;
use super::packed_candidate::MatmulPackedTuneCandidate;
use super::{floor, roofline};
use crate::tune::Predictor;
use cubek::matmul::kernels::{
    CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine, CubeStagedTmaFillerMatmulRoutine,
    CubeStagedTmaMatmulRoutine, CubeStagedTmaStoredMatmulRoutine, CubeStagedWgmmaMatmulRoutine,
    HypercubeLastArrivalMatmulRoutine, UnitRegistersMatmulRoutine,
};

/// The many-rows autotune's candidates on `hw`: the cube staged routine, the last-arrival hypercube
/// routine and the unit routine, in that order; against a packed input the staged routine reading
/// it as stored, in rows or in tiles, decoding it into its stages, the same with a packed
/// activation decoded first
/// ([`launch_activation_decoded`](MatmulPackedTuneCandidate::launch_activation_decoded)), and the
/// last-arrival routine reading it as stored, several cubes sharing each box along `K`.
///
/// The unit routine prepares everywhere, though a round orders the candidates by priority and its
/// small block often tiles a key the others mask. The last-arrival routine sizes its grid by the
/// device's workers and declines where it cannot.
pub(super) fn candidates(predictor: Predictor, groups: MatmulAutotuneGroups) -> MatmulTunables {
    let hw = predictor.hardware().clone();
    let mut autotune = candidate_list();
    let microkernel = autotune.group("microkernel", groups.many_rows);
    autotune
        .bounded(roofline::many_rows)
        .identified(
            &microkernel,
            MatmulTuneCandidate::list(CubeStagedMatmulRoutine, &predictor),
            RoutineIdentity::new,
            MatmulTuneCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulTuneCandidate::list(CubeStagedAsyncCopyMatmulRoutine, &predictor),
            RoutineIdentity::new,
            MatmulTuneCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulTuneCandidate::list(CubeStagedTmaMatmulRoutine, &predictor),
            RoutineIdentity::new,
            MatmulTuneCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulTuneCandidate::list(CubeStagedTmaFillerMatmulRoutine, &predictor),
            RoutineIdentity::new,
            MatmulTuneCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulTuneCandidate::list(CubeStagedTmaStoredMatmulRoutine, &predictor),
            RoutineIdentity::new,
            MatmulTuneCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulTuneCandidate::list(CubeStagedWgmmaMatmulRoutine, &predictor),
            RoutineIdentity::new,
            MatmulTuneCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulTuneCandidate::list(HypercubeLastArrivalMatmulRoutine, &predictor),
            RoutineIdentity::new,
            MatmulTuneCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulTuneCandidate::list(UnitRegistersMatmulRoutine, &predictor),
            RoutineIdentity::new,
            MatmulTuneCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulPackedTuneCandidate::of(
                CubeStagedMatmulRoutine,
                CubeStagedMatmulRoutine.packed_settings(&hw),
                &predictor,
            ),
            RoutineIdentity::new,
            MatmulPackedTuneCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulPackedTuneCandidate::of(
                CubeStagedAsyncCopyMatmulRoutine,
                CubeStagedAsyncCopyMatmulRoutine.packed_settings(&hw),
                &predictor,
            ),
            RoutineIdentity::new,
            MatmulPackedTuneCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulPackedTuneCandidate::of(
                CubeStagedTmaFillerMatmulRoutine,
                CubeStagedTmaFillerMatmulRoutine.packed_settings(&hw),
                &predictor,
            ),
            RoutineIdentity::new,
            MatmulPackedTuneCandidate::launch,
        )
        // The same, against a packed activation decoded first.
        .identified(
            &microkernel,
            MatmulPackedTuneCandidate::of(
                CubeStagedMatmulRoutine,
                CubeStagedMatmulRoutine.packed_settings(&hw),
                &predictor,
            ),
            MatmulPackedTuneCandidate::activation_decoded_identity,
            MatmulPackedTuneCandidate::launch_activation_decoded,
        )
        .identified(
            &microkernel,
            MatmulPackedTuneCandidate::of(
                CubeStagedAsyncCopyMatmulRoutine,
                CubeStagedAsyncCopyMatmulRoutine.packed_settings(&hw),
                &predictor,
            ),
            MatmulPackedTuneCandidate::activation_decoded_identity,
            MatmulPackedTuneCandidate::launch_activation_decoded,
        )
        .identified(
            &microkernel,
            MatmulPackedTuneCandidate::of(
                CubeStagedTmaFillerMatmulRoutine,
                CubeStagedTmaFillerMatmulRoutine.packed_settings(&hw),
                &predictor,
            ),
            MatmulPackedTuneCandidate::activation_decoded_identity,
            MatmulPackedTuneCandidate::launch_activation_decoded,
        )
        // The packed boxes again, several cubes sharing each along `K`: what a product with too
        // few boxes to fill the device and a long `K` wants.
        .identified(
            &microkernel,
            MatmulPackedTuneCandidate::of(
                HypercubeLastArrivalMatmulRoutine,
                HypercubeLastArrivalMatmulRoutine.packed_settings(&hw),
                &predictor,
            ),
            RoutineIdentity::new,
            MatmulPackedTuneCandidate::launch,
        )
        .floor(floor::decoded(&hw))
        .finish()
}
