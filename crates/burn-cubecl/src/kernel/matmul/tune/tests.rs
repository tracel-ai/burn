//! Tests for the matmul autotunes: how each candidate is prioritized and claims its launch, the
//! candidates every autotune holds, and what a race runs on.

use super::MatmulAutotuneGroups;
use super::candidate::{MaskedCells, MatmulTuneCandidate, shape_priority};
use super::packed_candidate::MatmulPackedTuneCandidate;
use super::{block_scaled, few_rows, many_rows};
use crate::tune::{CandidatePriority, TuneCandidate};
use cubecl::features::MmaConfig;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::quant::scheme::{BlockScale, QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::tile::Dimension;
use cubek::matmul::kernels::select::problem::OperandKind;
use cubek::matmul::kernels::{
    CubeBlock, CubeShuffleMatmulRoutine, CubeStagedMatmulRoutine, MatmulBatching, MatmulTuneKey,
    OperandKey, PackedContraction, PlaneShuffleMatmulRoutine,
};
use cubek::std::matmul_dimensions::{K, N};
use cubek::std::{Hardware, devices};

const F16: ElemType = ElemType::Float(FloatKind::F16);
const F32: ElemType = ElemType::Float(FloatKind::F32);

/// `hw` offering the `16×16×16` `f16` instruction summed in `f32`.
fn offering(hw: Hardware) -> Hardware {
    devices::offering_cmma(
        hw,
        [MmaConfig {
            a_type: F16,
            b_type: F16,
            cd_type: F32,
            m: 16,
            n: 16,
            k: 16,
        }],
    )
}

/// A key over an `f16` `m × k @ k × n` product of `batch` batches, row-major.
fn key(batch: usize, m: usize, n: usize, k: usize) -> MatmulTuneKey {
    MatmulTuneKey::new(
        F16,
        k,
        n,
        m,
        batch,
        OperandKey::float(Some(K), MatmulBatching::new(&[batch])),
        OperandKey::float(Some(N), MatmulBatching::new(&[batch])),
    )
}

/// A setting the device declines is tried after every candidate that runs, and claims no
/// launch an earlier candidate could duplicate: the cube routine's planes splitting `K`, on a
/// device whose shared memory takes no atomic add.
#[test]
fn a_declined_setting_is_tried_last_and_claims_no_launch() {
    let mut hw = offering(devices::strix_halo());
    devices::features(&mut hw).types.shared_atomic.clear();
    let key = key(1, 2048, 2048, 2048);
    let split = MatmulTuneCandidate::list(CubeStagedMatmulRoutine, &hw)
        .into_iter()
        .filter(|candidate| candidate.strategy().microkernel.planes_along_k.get() > 1);
    for candidate in split {
        let settings = candidate.strategy();
        assert_eq!(
            candidate.priority(&key),
            CandidatePriority::LAST,
            "{settings}"
        );
        assert_eq!(candidate.blueprint(&key), None, "{settings}");
    }
}

/// The priority is the autotune's one rule over the cube's output block: first where the block
/// tiles the product, last where one cube covers all of it.
#[test]
fn the_priority_reads_the_block_one_cube_covers() {
    let hw = offering(devices::strix_halo());
    let lead = MatmulTuneCandidate::list(CubeStagedMatmulRoutine, &hw).swap_remove(0);
    assert_eq!(
        lead.priority(&key(1, 2048, 2048, 2048)),
        CandidatePriority::FIRST
    );
    assert_eq!(
        lead.priority(&key(1, 16, 16, 2048)),
        CandidatePriority::LAST
    );
}

/// A grid masking rows costs the arithmetic it spends on them alone, so it is lowered where
/// that arithmetic is the units' own, and goes first beside the tiled grids where a tensor core
/// spends it: a decode's eight rows on the cube routine, whose block holds a tile's sixteen.
#[test]
fn masked_cells_cost_only_off_a_tensor_core() {
    let decode = key(1, 8, 4096, 4096);
    let masking = CubeBlock {
        m: 64,
        n: 64,
        k: 4096,
    };
    assert_eq!(
        shape_priority(&decode.shape(), masking, MaskedCells::Free),
        CandidatePriority::FIRST
    );
    assert!(shape_priority(&decode.shape(), masking, MaskedCells::Cost) < CandidatePriority::FIRST);

    let hw = offering(devices::strix_halo());
    let lead = MatmulTuneCandidate::list(CubeStagedMatmulRoutine, &hw).swap_remove(0);
    assert_eq!(lead.priority(&decode), CandidatePriority::FIRST);
}

/// On devices a dev machine cannot show: a routine that declines a device is tried last and
/// claims no launch, and one that prepares is placed by its block.
#[test]
fn a_decline_claims_nothing_on_every_device() {
    let key = key(1, 2048, 2048, 2048);
    for hw in [
        offering(devices::rdna_range()),
        offering(devices::cdna_wave64()),
        devices::wgpu_no_subgroups(),
        devices::cpu_12(),
    ] {
        let placed = |candidate_priority: CandidatePriority, claims: bool| match claims {
            true => assert_eq!(candidate_priority, CandidatePriority::FIRST),
            false => assert_eq!(candidate_priority, CandidatePriority::LAST),
        };
        for cube in MatmulTuneCandidate::list(CubeStagedMatmulRoutine, &hw) {
            placed(cube.priority(&key), cube.blueprint(&key).is_some());
        }
    }
}

/// The few-rows autotune's candidates: the shuffle routines, every packed rhs launched as
/// stored.
mod few_rows_candidates {
    use super::*;

    /// A key over an `f16` product of `batch` batches of `m` rows against a `k × n` rhs whose
    /// buffer runs along `rhs`, or along neither dimension for `None`.
    fn key(batch: usize, m: usize, n: usize, k: usize, rhs: Option<Dimension>) -> MatmulTuneKey {
        MatmulTuneKey::new(
            F16,
            k,
            n,
            m,
            batch,
            OperandKey::float(Some(K), MatmulBatching::new(&[batch])),
            OperandKey::float(rhs, MatmulBatching::Shared),
        )
    }

    /// A key over `m` rows of `f16` against a `k × n` q4 rhs packed along `K`.
    fn quantized_key(m: usize, n: usize, k: usize) -> MatmulTuneKey {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q4S)
            .with_levels([BlockScale::per_block([32, 1], ScaleDtype::F16)])
            .with_store(QuantStore::PackedU32(1));
        MatmulTuneKey::new(
            F16,
            k,
            n,
            m,
            1,
            OperandKey::float(Some(K), MatmulBatching::Shared),
            OperandKey {
                kind: OperandKind::Quantized(scheme),
                ..OperandKey::float(Some(K), MatmulBatching::Shared)
            },
        )
    }

    /// Where no shuffle routine serves such a product, or the device runs none of it, a
    /// candidate is tried last and claims no launch: a quantized rhs, an rhs lined along
    /// neither dimension, more rows than a work item holds, a device that commits to no plane
    /// width.
    #[test]
    fn a_candidate_serving_nothing_is_tried_last_and_claims_no_launch() {
        let cases = [
            (devices::strix_halo(), quantized_key(1, 2048, 2048)),
            (devices::strix_halo(), key(1, 1, 2048, 2048, None)),
            (devices::strix_halo(), key(1, 16, 2048, 2048, Some(K))),
            (devices::wgpu_no_subgroups(), key(1, 1, 2048, 2048, Some(K))),
        ];
        for (hw, key) in cases {
            for plane in MatmulTuneCandidate::list(PlaneShuffleMatmulRoutine, &hw) {
                let settings = plane.strategy();
                assert_eq!(plane.priority(&key), CandidatePriority::LAST, "{settings}");
                assert_eq!(plane.blueprint(&key), None, "{settings}");
            }
            for cube in MatmulTuneCandidate::list(CubeShuffleMatmulRoutine, &hw) {
                let settings = cube.strategy();
                assert_eq!(cube.priority(&key), CandidatePriority::LAST, "{settings}");
                assert_eq!(cube.blueprint(&key), None, "{settings}");
            }
        }
    }

    /// A candidate is placed by the one rule: a cube's run of columns tiles a decode
    /// projection, so both routines are tried first.
    #[test]
    fn the_priority_reads_the_block_one_cube_covers() {
        let hw = devices::strix_halo();
        let key = key(1, 1, 2048, 2048, Some(K));
        let plane = MatmulTuneCandidate::list(PlaneShuffleMatmulRoutine, &hw).swap_remove(0);
        let cube = MatmulTuneCandidate::list(CubeShuffleMatmulRoutine, &hw).swap_remove(0);
        assert_eq!(plane.priority(&key), CandidatePriority::FIRST);
        assert_eq!(cube.priority(&key), CandidatePriority::FIRST);
    }
}

/// At one row a packed key tries the settings in registers first and those on matrix
/// instructions after them; past one row it tries every packed setting first. A float key
/// tries them last.
#[test]
fn one_row_tries_the_registers_before_the_matrix_instructions() {
    let scheme = QuantScheme::default()
        .with_value(QuantValue::E2M1)
        .with_levels([BlockScale::per_block([16, 1], ScaleDtype::F16)])
        .with_store(QuantStore::PackedU32(1));
    let key = |rows, rhs: OperandKind| {
        MatmulTuneKey::new(
            F16,
            5120,
            5120,
            rows,
            1,
            OperandKey::float(Some(K), MatmulBatching::Shared),
            OperandKey {
                kind: rhs,
                ..OperandKey::float(Some(K), MatmulBatching::Shared)
            },
        )
    };
    let packed = OperandKind::Quantized(scheme);
    // The instruction a plane decodes a packed rhs into: f16 summed in f32, sixteen by eight.
    let hw = devices::offering_mma(
        devices::h100(),
        [MmaConfig {
            a_type: F16,
            b_type: F16,
            cd_type: F32,
            m: 16,
            n: 8,
            k: 16,
        }],
    );
    let settings = CubeShuffleMatmulRoutine::packed_settings(&hw);
    assert!(
        settings
            .iter()
            .any(PackedContraction::on_matrix_instructions),
        "the device lists settings on matrix instructions"
    );
    for candidate in MatmulPackedTuneCandidate::of(CubeShuffleMatmulRoutine, settings, &hw) {
        let one_row = match candidate.strategy().on_matrix_instructions() {
            true => CandidatePriority::LAST,
            false => CandidatePriority::FIRST,
        };
        let name = candidate.strategy();
        assert_eq!(candidate.priority(&key(1, packed)), one_row, "{name}");
        assert_eq!(
            candidate.priority(&key(8, packed)),
            CandidatePriority::FIRST,
            "{name}"
        );
        assert_eq!(
            candidate.priority(&key(8, OperandKind::Float)),
            CandidatePriority::LAST,
            "{name}"
        );
    }
}

/// Every autotune on every device holds the candidates `autotunes.txt` records: as many, under the
/// same names, which is what a persisted winner is checksummed by.
///
/// A line that moves is an autotune that races a candidate it did not, or stops racing one it did,
/// and every cached measurement of it is taken again: a change to an autotune rewrites the file on
/// purpose. Rewrite it with `CUBEK_WRITE_PLANS=1` and review the diff.
#[test]
fn every_table_holds_the_candidates_written_down() {
    let written = render_all();
    if std::env::var_os("CUBEK_WRITE_PLANS").is_some() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/kernel/matmul/tune/autotunes.txt"
        );
        std::fs::write(path, &written).expect("autotunes.txt is writable");
        return;
    }
    assert_eq!(
        written,
        include_str!("autotunes.txt").replace("\r\n", "\n"),
        "an autotune changed; rewrite autotunes.txt with CUBEK_WRITE_PLANS=1 and review the diff"
    );
}

/// Every autotune on every device, one line each: its candidate count and its names' checksum.
fn render_all() -> String {
    let groups = MatmulAutotuneGroups::default();
    let mut out = String::new();
    for (device, hw) in devices_listed() {
        out.push_str(&format!("# {device}\n"));
        for (autotune, built) in [
            ("few-rows", few_rows::candidates(hw.clone(), groups)),
            ("many-rows", many_rows::candidates(hw.clone(), groups)),
            ("block-scaled", block_scaled::candidates(hw.clone(), groups)),
        ] {
            out.push_str(&format!(
                "{autotune}: {} candidates, checksum {}\n",
                built.len(),
                built.compute_checksum()
            ));
        }
        out.push('\n');
    }
    out
}

/// The devices whose settings lists differ: a Mac, a datacenter GPU with and without the 4-bit
/// instruction, and a device without planes.
fn devices_listed() -> [(&'static str, Hardware); 4] {
    [
        ("m2-pro", devices::m2_pro()),
        ("h100", devices::h100()),
        ("h100 with nvfp4", devices::offering_nvfp4(devices::h100())),
        ("wgpu without subgroups", devices::wgpu_no_subgroups()),
    ]
}

/// **A race runs on the call's buffers.** The few-rows autotune hands its candidates the call's own
/// lhs and rhs, and a destination the call allocated once at the product's shape, which the winner
/// then writes too.
#[test]
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
fn a_race_runs_on_the_calls_buffers() {
    use crate::CubeDevice;
    use cubecl::CubeTuneId;
    use cubek::matmul::kernels::plan::{MatmulCall, MatmulCallInput};
    use cubek::matmul::kernels::tensor::{PlainTensor, StoredRead};

    let device = CubeDevice::default();
    let client = device.client();
    let input = |shape: [usize; 2]| {
        MatmulCallInput::plain(PlainTensor::zeros(&client, &shape, [shape[1], 1], F32))
    };
    let call = MatmulCall {
        hw: Hardware::from_client(&client),
        client: client.clone(),
        tune_id: CubeTuneId::new(&client, &device),
        lhs: input([3, 5]),
        rhs: input([5, 4]),
        out: F32,
        destination: None,
    }
    .with_destination();
    let destination = call.destination.clone().expect("a destination allocated");
    assert_eq!(destination.shape().as_slice(), [3, 4]);

    let tunables = few_rows::candidates(call.hw.clone(), MatmulAutotuneGroups::default());
    let raced = tunables.generate_inputs(&call.key(), &call);
    let memory = |input: &MatmulCallInput| match input.tensor.read() {
        StoredRead::Float => input.tensor.values().binding.handle.memory.id(),
        StoredRead::ScalesAlong(_) | StoredRead::Blocks(_) => unreachable!("a float call"),
    };
    assert_eq!(memory(&raced.lhs), memory(&call.lhs));
    assert_eq!(memory(&raced.rhs), memory(&call.rhs));
    let raced_destination = raced.destination.expect("the call's destination");
    assert!(matches!(
        raced_destination.handle.overlap(&destination.handle),
        cubecl::server::Overlap::Same
    ));
}
