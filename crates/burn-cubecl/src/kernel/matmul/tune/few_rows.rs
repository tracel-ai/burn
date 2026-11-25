use super::base::{MatmulAutotuneGroups, MatmulTunables, RoutineIdentity, candidate_list};
use super::candidate::MatmulTuneCandidate;
use super::packed_candidate::MatmulPackedTuneCandidate;
use super::{floor, roofline};
use crate::tune::Predictor;
use cubek::matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeStagedMatmulRoutine, PlaneShuffleMatmulRoutine,
    UnitWholeKMatmulRoutine,
};

/// The few-rows autotune's candidates on `hw`. Against float inputs the plane and cube shuffles and
/// the unit whole-K routine at their own settings; against a packed input the shuffles at the
/// settings worth trying against it, reading it as stored, and the packed staged routine at a
/// decode's settings; against a packed activation, the same again with the activation decoded first
/// ([`launch_activation_decoded`](MatmulPackedTuneCandidate::launch_activation_decoded)).
///
/// Keyed by the power of two at or above the call's rows, as the many-rows autotune is: a routine
/// that does not stage holds that many rows whatever the call's count inside it, so it serves or
/// declines every count of a key alike, and the one kernel it compiles serves them all.
pub(super) fn candidates(predictor: Predictor, groups: MatmulAutotuneGroups) -> MatmulTunables {
    let hw = predictor.hardware().clone();
    let mut autotune = candidate_list();
    let microkernel = autotune.group("microkernel", groups.few_rows);
    autotune
        .bounded(roofline::few_rows)
        .identified(
            &microkernel,
            MatmulTuneCandidate::list(PlaneShuffleMatmulRoutine, &predictor),
            RoutineIdentity::new,
            MatmulTuneCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulTuneCandidate::list(CubeShuffleMatmulRoutine, &predictor),
            RoutineIdentity::new,
            MatmulTuneCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulTuneCandidate::list(UnitWholeKMatmulRoutine, &predictor),
            RoutineIdentity::new,
            MatmulTuneCandidate::launch,
        )
        // Against a packed input, read as stored, in rows or in its tile order. The plane routine
        // leads, so a tie resolves to what a pin runs.
        .identified(
            &microkernel,
            MatmulPackedTuneCandidate::of(
                PlaneShuffleMatmulRoutine,
                PlaneShuffleMatmulRoutine::packed_settings(),
                &hw,
            ),
            RoutineIdentity::new,
            MatmulPackedTuneCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulPackedTuneCandidate::of(
                CubeShuffleMatmulRoutine,
                CubeShuffleMatmulRoutine::packed_settings(&hw),
                &hw,
            ),
            RoutineIdentity::new,
            MatmulPackedTuneCandidate::launch,
        )
        // The packed staged routine at a decode's settings: each packed input decoded into its
        // stages and contracted on the tensor cores, which the shuffles' software sums cannot keep
        // up with past a few rows.
        .identified(
            &microkernel,
            MatmulPackedTuneCandidate::of(
                CubeStagedMatmulRoutine,
                CubeStagedMatmulRoutine.packed_one_step_settings(),
                &hw,
            ),
            RoutineIdentity::new,
            MatmulPackedTuneCandidate::launch,
        )
        // The same, against a packed activation decoded first.
        .identified(
            &microkernel,
            MatmulPackedTuneCandidate::of(
                PlaneShuffleMatmulRoutine,
                PlaneShuffleMatmulRoutine::packed_settings(),
                &hw,
            ),
            MatmulPackedTuneCandidate::activation_decoded_identity,
            MatmulPackedTuneCandidate::launch_activation_decoded,
        )
        .identified(
            &microkernel,
            MatmulPackedTuneCandidate::of(
                CubeShuffleMatmulRoutine,
                CubeShuffleMatmulRoutine::packed_settings(&hw),
                &hw,
            ),
            MatmulPackedTuneCandidate::activation_decoded_identity,
            MatmulPackedTuneCandidate::launch_activation_decoded,
        )
        .identified(
            &microkernel,
            MatmulPackedTuneCandidate::of(
                CubeStagedMatmulRoutine,
                CubeStagedMatmulRoutine.packed_one_step_settings(),
                &hw,
            ),
            MatmulPackedTuneCandidate::activation_decoded_identity,
            MatmulPackedTuneCandidate::launch_activation_decoded,
        )
        .floor(floor::decoded(&hw))
        .finish()
}
