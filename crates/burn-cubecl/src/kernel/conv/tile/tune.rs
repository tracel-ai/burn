//! The tile convolution's autotune: every setting the cube staged routine lists on the device
//! ([`CubeStagedConvCandidate::list`]), raced per problem, over the first of them that serves and
//! the plane window routine as the floor.

use crate::tune::{
    Autotune, CandidateIdentity, CandidatePriority, Ceiling, GroupPriority, Identity, Roofline,
    TuneCandidate,
};
use cubecl::client::Client;
use cubecl::throughput::{CmmaDims, MemoryAccess, ThroughputKey};
use cubecl::tile::launch::Decline;
use cubecl::tune::{LocalTuner, TunableSet, local_tuner};
use cubecl_common::device::ServiceId;
use cubek::convolution::kernels::tiled::{
    ConvCoverage, ConvRoutineKind, ConvTensors, CubeStagedConvCandidate, CubeStagedConvPrepared,
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
        let (list, hw) = (list.clone(), hw.clone());
        move || candidates(list.clone(), hw.clone())
    });
    let key = CubeStagedConvTuneKey::of(&tensors, &hw);
    let serves = |candidates: &mut dyn Iterator<Item = &CubeStagedConvCandidate>| {
        let attempts = candidates.map(|candidate| candidate.serves(&tensors));
        first_served(attempts, ROUTINE_LISTS_NO_SETTING).is_ok()
    };
    let tuned = match TUNER.fastest_identity(&id, &tunables, &key) {
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
#[derive(Debug, Clone)]
struct TileConvTuneCandidate {
    candidate: CubeStagedConvCandidate,
    hw: Hardware,
}

impl TuneCandidate for TileConvTuneCandidate {
    type Strategy = CubeStagedConvSettings;
    type Key = CubeStagedConvTuneKey;
    type Blueprint = CubeStagedConvPrepared;

    fn strategy(&self) -> &CubeStagedConvSettings {
        self.candidate.settings()
    }

    /// As [`priority`] judges the launch the candidate prepares; last where the routine declines
    /// the problem, which it then does at launch before compiling anything.
    fn priority(&self, key: &CubeStagedConvTuneKey) -> CandidatePriority {
        match self.candidate.prepared(key) {
            Some(prepared) => priority(
                prepared.coverage(),
                prepared.cube_dim().num_elems(),
                &self.hw,
            ),
            None => CandidatePriority::LAST,
        }
    }

    fn blueprint(&self, key: &CubeStagedConvTuneKey) -> Option<CubeStagedConvPrepared> {
        self.candidate.prepared(key)
    }
}

/// Where a candidate whose launch lies over its problem as `coverage`, in cubes of
/// `units_per_cube` units, is tried on `hw`: the top band unless its boxes cover at least twice
/// the problem or its grid fills less than half the device, then one band lower. A band is raced
/// whole and a lower one only when it measures nothing, so the priority only separates the
/// clearly wasteful from the rest.
fn priority(coverage: &ConvCoverage, units_per_cube: u32, hw: &Hardware) -> CandidatePriority {
    let wasteful = coverage.covered_cells >= 2 * coverage.problem_cells.max(1);
    let idle = hw
        .cubes_to_fill(units_per_cube)
        .is_some_and(|wanted| wanted as usize > 2 * coverage.cube_count.max(1));
    match wasteful || idle {
        true => CandidatePriority::FIRST.lowered(1),
        false => CandidatePriority::FIRST,
    }
}

/// The bound `key`'s convolution has on `hw`: the largest instruction the device offers at the
/// key's elements accumulating in the key's sum (plain arithmetic where it offers none), or the
/// read bus, whichever is higher for the key.
fn ceiling(key: &CubeStagedConvTuneKey, hw: &Hardware) -> Ceiling {
    let instruction = hw
        .properties
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
        });
    Ceiling::Either {
        compute: ThroughputKey::compute(instruction, key.lhs, key.sum),
        memory: MemoryAccess::Read,
    }
}

/// The autotune over `list` on `hw`: every listed setting, a round stopping at the first within
/// reach of the bound the convolution has, and [`launch_floor`] where none serves.
fn candidates(list: Arc<[CubeStagedConvCandidate]>, hw: Hardware) -> TileConvTunables {
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
            hw: hw.clone(),
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

    /// The units of a cube of four planes of 32.
    const UNITS_PER_CUBE: u32 = 128;
    /// Cells a problem has, which a close launch covers.
    const PROBLEM_CELLS: usize = 1 << 20;

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

    /// The cubes that fill `hw` in cubes of [`UNITS_PER_CUBE`] units.
    fn filling(hw: &Hardware) -> usize {
        hw.cubes_to_fill(UNITS_PER_CUBE)
            .expect("the device states its cores") as usize
    }

    /// A launch covering `covered_cells` of the problem's cells in `cube_count` cubes.
    fn coverage(covered_cells: usize, cube_count: usize) -> ConvCoverage {
        ConvCoverage {
            covered_cells,
            problem_cells: PROBLEM_CELLS,
            cube_count,
        }
    }

    /// Boxes overhanging the problem a little, on a grid that fills the device, are raced in the
    /// top band.
    #[test]
    fn a_close_launch_filling_the_device_is_first() {
        let hw = nvidia();
        let close = coverage(PROBLEM_CELLS + PROBLEM_CELLS / 8, filling(&hw));
        assert_eq!(
            priority(&close, UNITS_PER_CUBE, &hw),
            CandidatePriority::FIRST
        );
    }

    /// Boxes covering at least twice the problem are raced a band lower.
    #[test]
    fn boxes_covering_twice_the_problem_are_a_band_lower() {
        let hw = nvidia();
        let overhanging = coverage(2 * PROBLEM_CELLS, filling(&hw));
        assert_eq!(
            priority(&overhanging, UNITS_PER_CUBE, &hw),
            CandidatePriority::FIRST.lowered(1)
        );
    }

    /// A grid short of half the cubes that fill the device is raced a band lower; one of half
    /// of them is not.
    #[test]
    fn a_grid_short_of_half_the_device_is_a_band_lower() {
        let hw = nvidia();
        let at = |cube_count| priority(&coverage(PROBLEM_CELLS, cube_count), UNITS_PER_CUBE, &hw);
        let half = filling(&hw).div_ceil(2);
        assert_eq!(at(half - 1), CandidatePriority::FIRST.lowered(1));
        assert_eq!(at(half), CandidatePriority::FIRST);
    }

    /// The set raced over the device's list comes out the same every time it is built, so the
    /// autotune's checksum is stable.
    #[test]
    fn the_set_is_built_in_one_order() {
        let hw = nvidia();
        let checksum = || {
            let list = CubeStagedConvCandidate::list(&hw).into();
            candidates(list, hw.clone()).compute_checksum()
        };
        assert!(!CubeStagedConvCandidate::list(&hw).is_empty());
        assert_eq!(checksum(), checksum());
    }
}
