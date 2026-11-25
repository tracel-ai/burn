//! The reduce routines' autotune: every routine at every setting the device lists
//! ([`ReduceCandidate::list`]), raced per problem, over the unit routine as the floor.

use crate::tune::{
    Arithmetic, Autotune, CandidateIdentity, CandidatePriority, Ceiling, GroupPriority, Identity,
    Predictor, Roofline, TuneCandidate,
};
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::throughput::MemoryAccess;
use cubecl::tune::{LocalTuner, TunableSet, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek::reduce::kernels::{
    Reduce, ReduceCandidate, ReduceKernel, ReduceKey, ReduceOpStrategy, ReducePrepared,
    ReduceTensors,
};
use cubek::std::routine::EstimatedProblem;
use std::sync::Arc;

/// What a candidate of the autotune is identified by: the candidate itself, or the floor by name.
type ReduceIdentity = CandidateIdentity<ReduceCandidate>;

/// The autotune as it is built, before it closes into its candidates.
type ReduceAutotune = Autotune<ReduceKey, ReduceTensors, (), ReduceIdentity>;

/// The autotune's candidates, as the tuner holds them.
type ReduceTunables = Arc<TunableSet<ReduceKey, ReduceTensors, (), ReduceIdentity>>;

static TUNER: LocalTuner<ReduceKey, CubeTuneId> = local_tuner!("reduce");

/// `tensors` reduced by the routine the autotune finds fastest for them on `device`. It never
/// refuses: the floor serves every problem.
pub(crate) fn launch_tuned(tensors: ReduceTensors, device: &Device) {
    let client = tensors.client().clone();
    let id = CubeTuneId::new(&client, device);
    let tunables = tunables(&id, &client);
    TUNER.execute(&id, &client, tunables, tensors);
}

/// The candidate the tuner on `client`'s `device` found fastest for `key`, or `None` where no
/// reduction in this process has settled the key, by a round or from the winner one left on disk,
/// or the floor named apart won it: what a fused reduce runs, so it inherits the kernel the unfused
/// round chose.
#[cfg(feature = "fusion")]
pub(crate) fn fastest_tuned(
    client: &Client,
    device: &Device,
    key: &ReduceKey,
) -> Option<ReduceCandidate> {
    let id = CubeTuneId::new(client, device);
    let tunables = tunables(&id, client);
    TUNER.fastest_identity(&id, &tunables, key)?.routine()
}

/// The autotune for the device `id` names, built on first touch. The one place it is built: a
/// tuner caches a set per initializer *type*, so a second `init` site would build a second set.
fn tunables(id: &CubeTuneId, client: &Client) -> ReduceTunables {
    let predictor = Predictor::on_device(client);
    TUNER.init(id, move || candidates(predictor.clone()))
}

/// One routine at one of its settings, as the autotune races it, on the device it was listed for.
#[derive(Debug, Clone)]
struct ReduceTuneCandidate {
    candidate: ReduceCandidate,
    predictor: Predictor,
}

impl TuneCandidate for ReduceTuneCandidate {
    type Strategy = String;
    type Key = ReduceKey;
    type Blueprint = ReducePrepared;

    fn strategy(&self) -> &String {
        self.candidate.name()
    }

    /// The routine's estimate at this setting, timed at the device's peaks, against the
    /// reduction's roofline, the arithmetic counted in `f32`.
    fn priority(&self, key: &ReduceKey) -> CandidatePriority {
        self.predictor.priority_of(
            self.candidate.estimate(key),
            Arithmetic::plain(ElemType::Float(FloatKind::F32)),
        )
    }

    fn blueprint(&self, key: &ReduceKey) -> Option<ReducePrepared> {
        self.candidate.prepared(key)
    }
}

/// Every routine's candidates on `hw`, identified by their routine and setting, over the unit
/// routine as the floor, which serves every problem on every device.
fn candidates(predictor: Predictor) -> TunableSet<ReduceKey, ReduceTensors, (), ReduceIdentity> {
    let mut autotune = ReduceAutotune::new(|tensors: &ReduceTensors| ReduceKey::from(tensors));
    let tile = autotune.group("tile", GroupPriority::DEFAULT);
    let raced = ReduceCandidate::list(predictor.hardware())
        .into_iter()
        .map(|candidate| ReduceTuneCandidate {
            candidate,
            predictor: predictor.clone(),
        })
        .collect();
    autotune
        .identified(
            &tile,
            raced,
            |raced: &ReduceTuneCandidate| ReduceIdentity::Routine(raced.candidate.clone()),
            |raced, tensors: ReduceTensors| {
                raced
                    .candidate
                    .launch(&tensors)
                    .map_err(|decline| decline.to_string())
            },
        )
        .bounded(|_, tensors: &ReduceTensors| Roofline {
            client: tensors.client().clone(),
            ceiling: Ceiling::Memory(MemoryAccess::Read),
            work: tensors.problem().demand(),
        })
        .floor(ReduceIdentity::named(
            "unit-direct-reduce floor",
            |tensors: ReduceTensors| {
                Reduce::new(ReduceOpStrategy::Specific(ReduceKernel::Unit))
                    .launch(tensors)
                    .map_err(|error| error.to_string())
            },
        ))
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::tile::Geometry;
    use cubek::reduce::ReduceOperation;
    use cubek::reduce::kernels::{ReduceElements, ReduceRoutineProblem};
    use cubek::std::devices;

    const F32: ElemType = ElemType::Float(FloatKind::F32);
    const ROW_LEN: usize = 16;

    /// The key of sums of `rows` rows of [`ROW_LEN`] values, in `elements`.
    fn key_of_rows_in(rows: usize, elements: ReduceElements) -> ReduceKey {
        ReduceKey::from(&ReduceRoutineProblem::stated(
            &Geometry::row_major(&[rows, ROW_LEN]),
            &Geometry::row_major(&[rows, 1]),
            1,
            ReduceOperation::Sum,
            elements,
        ))
    }

    /// The key of sums of `rows` rows of [`ROW_LEN`] values, in f32.
    fn key_of_rows(rows: usize) -> ReduceKey {
        key_of_rows_in(
            rows,
            ReduceElements::for_operation(ReduceOperation::Sum, F32, None),
        )
    }

    /// The first candidate on `predictor`'s device whose name starts with `routine`.
    fn raced(routine: &str, predictor: Predictor) -> ReduceTuneCandidate {
        let candidate = ReduceCandidate::list(predictor.hardware())
            .into_iter()
            .find(|candidate| candidate.name().starts_with(routine))
            .expect("the routine lists a setting on every device");
        ReduceTuneCandidate {
            candidate,
            predictor,
        }
    }

    /// A candidate that prepares on the device is placed by its prediction, which never puts it
    /// with the declines.
    #[test]
    fn a_candidate_that_prepares_is_placed_above_the_declines() {
        let unit = raced(
            "unit-direct-reduce",
            Predictor::on_device(&cubecl::test_device().client()),
        );
        for rows in [8, 1 << 18] {
            assert!(unit.priority(&key_of_rows(rows)) > CandidatePriority::LAST);
        }
    }

    #[test]
    fn a_candidate_that_declines_the_key_comes_last() {
        let declining = raced(
            "cube-staged-reduce",
            Predictor::described(devices::lacking_every_feature(devices::m2_pro())),
        );
        assert_eq!(declining.priority(&key_of_rows(8)), CandidatePriority::LAST);
    }

    /// A sum no caller asks for: it accumulates in less than it reads.
    #[cfg(feature = "fusion")]
    #[test]
    fn a_key_no_reduction_settled_has_no_fastest_candidate() {
        let device = cubecl::test_device();
        let key = key_of_rows_in(
            4,
            ReduceElements {
                input: ElemType::Float(FloatKind::F64),
                accumulator: ElemType::Float(FloatKind::F16),
                out: F32,
            },
        );
        assert!(fastest_tuned(&device.client(), &device, &key).is_none());
    }
}
