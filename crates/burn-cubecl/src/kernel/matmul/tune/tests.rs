//! Tests for the matmul autotunes: how each candidate is prioritized and claims its launch, the
//! candidates every autotune holds, and what a race runs on.

use super::MatmulAutotuneGroups;
use super::candidate::MatmulTuneCandidate;
use super::packed_candidate::MatmulPackedTuneCandidate;
use super::{block_scaled, few_rows, many_rows};
use crate::tune::{CandidatePriority, Predictor, TuneCandidate};
use cubecl::features::MmaConfig;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::tile::Dimension;
use cubek::matmul::kernels::select::problem::OperandKind;
use cubek::matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeStagedMatmulRoutine, MatmulBatching, MatmulPackedCandidate,
    MatmulTuneKey, OperandKey, PlaneShuffleMatmulRoutine,
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
    let split =
        MatmulTuneCandidate::list(CubeStagedMatmulRoutine, &Predictor::described(hw.clone()))
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

/// A grid that leaves the device idle is predicted further above its roofline than one that
/// fills it: the lead setting on a product a single cube covers, against one that spreads over
/// every worker.
#[test]
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
fn a_grid_that_leaves_the_device_idle_is_tried_later() {
    let predictor = Predictor::on_device(&crate::CubeDevice::default().client());
    let lead = MatmulTuneCandidate::list(CubeStagedMatmulRoutine, &predictor).swap_remove(0);
    assert!(lead.priority(&key(1, 16, 16, 2048)) < lead.priority(&key(1, 2048, 2048, 2048)));
}

/// On devices a dev machine cannot show: a routine that declines a device is tried last and
/// claims no launch, and one that prepares is first, there being no device to time it on.
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
        for cube in
            MatmulTuneCandidate::list(CubeStagedMatmulRoutine, &Predictor::described(hw.clone()))
        {
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
            .per_block([32, 1], ScaleDtype::F16)
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
            for plane in MatmulTuneCandidate::list(
                PlaneShuffleMatmulRoutine,
                &Predictor::described(hw.clone()),
            ) {
                let settings = plane.strategy();
                assert_eq!(plane.priority(&key), CandidatePriority::LAST, "{settings}");
                assert_eq!(plane.blueprint(&key), None, "{settings}");
            }
            for cube in MatmulTuneCandidate::list(
                CubeShuffleMatmulRoutine,
                &Predictor::described(hw.clone()),
            ) {
                let settings = cube.strategy();
                assert_eq!(cube.priority(&key), CandidatePriority::LAST, "{settings}");
                assert_eq!(cube.blueprint(&key), None, "{settings}");
            }
        }
    }

    /// Both shuffle routines prepare a decode projection, so with no device to time them on
    /// both are tried first.
    #[test]
    fn a_routine_that_prepares_is_first_without_a_device() {
        let hw = devices::strix_halo();
        let key = key(1, 1, 2048, 2048, Some(K));
        let plane =
            MatmulTuneCandidate::list(PlaneShuffleMatmulRoutine, &Predictor::described(hw.clone()))
                .swap_remove(0);
        let cube =
            MatmulTuneCandidate::list(CubeShuffleMatmulRoutine, &Predictor::described(hw.clone()))
                .swap_remove(0);
        assert_eq!(plane.priority(&key), CandidatePriority::FIRST);
        assert_eq!(cube.priority(&key), CandidatePriority::FIRST);
    }
}

/// A packed key of `rows` rows of f16 against a `5120 × 5120` rhs stored as `rhs` says.
fn packed_key(rows: usize, rhs: OperandKind) -> MatmulTuneKey {
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
}

/// A q4 rhs packed along `K` in rows, sixteen values a scale.
fn q4() -> OperandKind {
    OperandKind::Quantized(
        QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .per_block([16, 1], ScaleDtype::F16)
            .with_store(QuantStore::PackedU32(1)),
    )
}

/// A packed setting is tried last on a float key, which it declines before compiling anything;
/// on a packed one where there is no device to time it on, first where it prepares the problem
/// and last where it declines it.
#[test]
fn a_packed_setting_is_last_on_a_float_key_and_first_without_a_device() {
    let hw = devices::h100();
    let settings = CubeShuffleMatmulRoutine::packed_settings(&hw);
    let estimated = MatmulPackedCandidate::of(CubeShuffleMatmulRoutine, settings.clone(), &hw);
    let predictor = Predictor::described(hw);
    let packed = packed_key(8, q4());
    let candidates = MatmulPackedTuneCandidate::of(CubeShuffleMatmulRoutine, settings, &predictor);
    for (candidate, estimated) in candidates.iter().zip(&estimated) {
        let name = candidate.strategy();
        let expected = match estimated.estimate(&packed) {
            Some(_) => CandidatePriority::FIRST,
            None => CandidatePriority::LAST,
        };
        assert_eq!(candidate.priority(&packed), expected, "{name}");
        assert_eq!(
            candidate.priority(&packed_key(8, OperandKind::Float)),
            CandidatePriority::LAST,
            "{name}"
        );
    }
    assert!(
        estimated
            .iter()
            .any(|candidate| candidate.estimate(&packed).is_some()),
        "a packed setting the device lists prepares the projection"
    );
}

/// On a device, a packed setting is placed by the estimate of the launch it prepares for the
/// packed problem a key describes: one at least of the settings the device lists prepares a
/// decode step's q4 projection, and its prediction never puts it with the declines.
#[test]
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
fn a_packed_setting_is_placed_by_its_estimate_on_a_device() {
    let client = crate::CubeDevice::default().client();
    let predictor = Predictor::on_device(&client);
    let settings = CubeShuffleMatmulRoutine::packed_settings(predictor.hardware());
    let candidates = MatmulPackedTuneCandidate::of(CubeShuffleMatmulRoutine, settings, &predictor);
    let key = packed_key(1, q4());
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.priority(&key) > CandidatePriority::LAST),
        "a packed setting the device lists prepares the decode projection"
    );
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
            (
                "few-rows",
                few_rows::candidates(Predictor::described(hw.clone()), groups),
            ),
            (
                "many-rows",
                many_rows::candidates(Predictor::described(hw.clone()), groups),
            ),
            (
                "block-scaled",
                block_scaled::candidates(Predictor::described(hw.clone()), groups),
            ),
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
    use cubek::matmul::kernels::tensor::{PlainTensor, StoredTensor};

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

    let tunables = few_rows::candidates(
        Predictor::on_device(&call.client),
        MatmulAutotuneGroups::default(),
    );
    let raced = tunables.generate_inputs(&call.key(), &call);
    let memory = |input: &MatmulCallInput| match &input.tensor {
        StoredTensor::Plain(tensor) => tensor.binding.handle.memory.id(),
        StoredTensor::Scaled(_) | StoredTensor::Packed(_) => unreachable!("a float call"),
    };
    assert_eq!(memory(&raced.lhs), memory(&call.lhs));
    assert_eq!(memory(&raced.rhs), memory(&call.rhs));
    let raced_destination = raced.destination.expect("the call's destination");
    assert!(matches!(
        raced_destination.handle.overlap(&destination.handle),
        cubecl::server::Overlap::Same
    ));
}
