//! The tile convolution's autotune: every setting the cube staged routine lists on the device
//! ([`CubeStagedConvCandidate::list`]), raced per problem, over the first of them that serves and
//! the plane window routine as the floor.

use crate::tune::{
    Arithmetic, Autotune, CandidateIdentity, CandidatePriority, Ceiling, GroupPriority, Identity,
    Predictor, Roofline, TuneCandidate,
};
use cubecl::client::Client;
use cubecl::throughput::{CmmaDims, MemoryAccess, ThroughputKey};
use cubecl::tile::launch::Decline;
use cubecl::tune::{LocalTuner, TunableSet, local_tuner};
use cubecl_common::device::ServiceId;
use cubek::convolution::kernels::tiled::{
    ConvRoutineKind, ConvTensors, CubeStagedConvCandidate, CubeStagedConvPrepared,
    CubeStagedConvSettings, CubeStagedConvTuneKey,
};
use cubek::std::routine::{ROUTINE_LISTS_NO_SETTING, first_served};
use cubek::std::{Hardware, Instructions, manual_mma};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

const TILE_GROUP: &str = "tile";
const FLOOR_NAME: &str = "conv-tile-floor";

/// What a candidate of the autotune is identified by: its settings, or the floor by name.
type TileConvIdentity = CandidateIdentity<CubeStagedConvSettings>;

/// The autotune as it is built, before it closes into its candidates.
type TileConvAutotune = Autotune<CubeStagedConvTuneKey, ConvTensors, (), TileConvIdentity>;

/// The autotune's candidates.
type TileConvTunables = TunableSet<CubeStagedConvTuneKey, ConvTensors, (), TileConvIdentity>;

static TUNER: LocalTuner<CubeStagedConvTuneKey, ServiceId> = local_tuner!("tile-conv");

/// The list each device races, built once so every launch and the autotune share the launches its
/// candidates prepared.
static LISTS: LazyLock<Mutex<HashMap<ServiceId, Arc<[CubeStagedConvCandidate]>>>> =
    LazyLock::new(Mutex::default);

/// The convolution `tensors` state, at the setting the autotune finds fastest for it among the
/// cube staged routine's listed settings on its device.
///
/// # Guarantees
///
/// - Nothing is launched where the output is empty.
/// - Where no listed setting serves the problem, or the one the autotune settled for its key does
///   not, the first that does launches, and the plane window routine is the floor under them all.
///
/// # Errors
///
/// The plane window routine's refusal where nothing serves the problem.
pub fn conv_tile_autotune(tensors: ConvTensors) -> Result<(), Decline> {
    if tensors.is_empty() {
        return Ok(());
    }
    let client = tensors.client().clone();
    let hw = Hardware::from_client(&client);
    let id = client.service_id();
    let list = listed(&id, &hw);
    let tunables = TUNER.init(&id, {
        let (list, predictor) = (list.clone(), Predictor::on_device(&client));
        move || candidates(list.clone(), predictor.clone())
    });
    let key = CubeStagedConvTuneKey::of(&tensors, &hw);
    let serves = |candidates: &mut dyn Iterator<Item = &CubeStagedConvCandidate>| {
        let attempts = candidates.map(|candidate| candidate.serves(&tensors));
        first_served(attempts, ROUTINE_LISTS_NO_SETTING).is_ok()
    };
    let tuned = match TUNER.winner_identity(&id, &tunables, &key) {
        None => serves(&mut list.iter()),
        Some(winner) => match winner.routine() {
            Some(settings) => serves(&mut list.iter().filter(|c| c.settings() == &settings)),
            None => true,
        },
    };
    match tuned {
        true => run_tuned(&id, &client, tunables, tensors),
        false => launch_floor(&list, &hw, tensors),
    }
}

/// The list the device `id` names races, listed on `hw` on first touch.
fn listed(id: &ServiceId, hw: &Hardware) -> Arc<[CubeStagedConvCandidate]> {
    LISTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(*id)
        .or_insert_with(|| CubeStagedConvCandidate::list(hw).into())
        .clone()
}

/// `tensors` launched at the setting the autotune picks among `tunables`.
fn run_tuned(
    id: &ServiceId,
    client: &Client,
    tunables: Arc<TileConvTunables>,
    tensors: ConvTensors,
) -> Result<(), Decline> {
    TUNER.execute(id, client, tunables, tensors);
    Ok(())
}

/// `tensors` at the first of `list` that serves them, or on the plane window routine where none
/// does.
fn launch_floor(
    list: &[CubeStagedConvCandidate],
    hw: &Hardware,
    tensors: ConvTensors,
) -> Result<(), Decline> {
    let listed = list
        .iter()
        .map(|candidate| candidate.launch(tensors.clone()));
    let floor =
        std::iter::once_with(|| ConvRoutineKind::PlaneWindow.launch_on(hw, tensors.clone()));
    first_served(listed.chain(floor), ROUTINE_LISTS_NO_SETTING)
}

/// One listed setting, as the autotune races it, on the device it was listed for.
///
/// Its priority is its routine's estimate timed at the device's peaks, against the
/// convolution's roofline ([`Predictor::priority_of`]).
#[derive(Debug, Clone)]
struct TileConvTuneCandidate {
    candidate: CubeStagedConvCandidate,
    predictor: Predictor,
}

impl TuneCandidate for TileConvTuneCandidate {
    type Strategy = CubeStagedConvSettings;
    type Key = CubeStagedConvTuneKey;
    type Blueprint = CubeStagedConvPrepared;

    fn strategy(&self) -> &CubeStagedConvSettings {
        self.candidate.settings()
    }

    /// Its estimate timed at the device's peaks, on the largest instruction the device offers
    /// the key's elements; last where the routine declines the problem, which it then does at
    /// launch before compiling anything.
    fn priority(&self, key: &CubeStagedConvTuneKey) -> CandidatePriority {
        let hw = self.predictor.hardware();
        let arithmetic = Arithmetic {
            vector: ThroughputKey::compute(None, key.lhs, key.sum),
            matrix: instruction(key, hw)
                .map(|dims| ThroughputKey::compute(Some(dims), key.lhs, key.sum)),
        };
        self.predictor
            .priority_of(self.candidate.estimate(key), arithmetic)
    }

    fn blueprint(&self, key: &CubeStagedConvTuneKey) -> Option<CubeStagedConvPrepared> {
        self.candidate.prepared(key)
    }
}

/// The largest matrix instruction `hw` offers at `key`'s elements accumulating in the key's sum,
/// or `None` where it offers none.
fn instruction(key: &CubeStagedConvTuneKey, hw: &Hardware) -> Option<CmmaDims> {
    hw.properties
        .features
        .matmul
        .cmma
        .iter()
        .copied()
        .chain(manual_mma(hw))
        .for_triple(key.lhs, key.rhs, key.sum)
        .largest()
        .map(|shape| CmmaDims {
            m: shape.m as usize,
            n: shape.n as usize,
            k: shape.k as usize,
        })
}

/// The bound `key`'s convolution has on `hw`: the largest instruction the device offers at the
/// key's elements accumulating in the key's sum (plain arithmetic where it offers none), or the
/// read bus, whichever is higher for the key.
fn ceiling(key: &CubeStagedConvTuneKey, hw: &Hardware) -> Ceiling {
    Ceiling::Either {
        compute: ThroughputKey::compute(instruction(key, hw), key.lhs, key.sum),
        memory: MemoryAccess::Read,
    }
}

/// The autotune over `list` on `predictor`'s device: every listed setting, a round stopping at the first within
/// reach of the bound the convolution has, and [`launch_floor`] where none serves.
fn candidates(list: Arc<[CubeStagedConvCandidate]>, predictor: Predictor) -> TileConvTunables {
    let hw = predictor.hardware().clone();
    let keyed = hw.clone();
    let mut autotune = TileConvAutotune::new(move |tensors: &ConvTensors| {
        CubeStagedConvTuneKey::of(tensors, &keyed)
    });
    let tile = autotune.group(TILE_GROUP, GroupPriority::DEFAULT);
    let floor = (list.clone(), hw.clone());
    let raced = list
        .iter()
        .map(|candidate| TileConvTuneCandidate {
            candidate: candidate.clone(),
            predictor: predictor.clone(),
        })
        .collect();
    autotune
        .bounded(
            move |key: &CubeStagedConvTuneKey, tensors: &ConvTensors| Roofline {
                client: tensors.client().clone(),
                ceiling: ceiling(key, &hw),
                work: tensors.demand(),
            },
        )
        .identified(&tile, raced, CandidateIdentity::new, |raced, tensors| {
            raced.candidate.launch(tensors).map_err(|refusal| {
                format!(
                    "{} does not serve this problem: {refusal:?}",
                    raced.strategy()
                )
            })
        })
        .floor(<TileConvIdentity as Identity>::named(
            FLOOR_NAME,
            move |tensors: ConvTensors| {
                let (list, hw) = &floor;
                launch_floor(list, hw, tensors).map_err(|refusal| format!("{refusal:?}"))
            },
        ))
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek::std::devices;

    /// An NVIDIA GPU offering cmma `16x16x16` over `f16`, accumulating in `f32`.
    fn nvidia() -> Hardware {
        let f16 = ElemType::Float(FloatKind::F16);
        let shape = MmaConfig {
            a_type: f16,
            b_type: f16,
            cd_type: ElemType::Float(FloatKind::F32),
            m: 16,
            n: 16,
            k: 16,
        };
        devices::offering_cmma(devices::h100(), [shape])
    }

    /// The set raced over the device's list comes out the same every time it is built, so the
    /// autotune's checksum is stable.
    #[test]
    fn the_set_is_built_in_one_order() {
        let hw = nvidia();
        let checksum = || {
            let list = CubeStagedConvCandidate::list(&hw).into();
            candidates(list, Predictor::described(hw.clone())).compute_checksum()
        };
        assert!(!CubeStagedConvCandidate::list(&hw).is_empty());
        assert_eq!(checksum(), checksum());
    }
}
