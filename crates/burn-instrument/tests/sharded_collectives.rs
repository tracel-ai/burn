//! What an instrumented run on a sharded compute device of two cards records: one row per kind
//! of collective, each member's part of every kernel, each timed on its own card, and the lead's
//! floor and peaks.
#![cfg(any(feature = "sharded-tests-cuda", feature = "sharded-tests-rocm"))]

use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use burn_instrument::{Bounds, CollectivePath, Instrument, Peaks, compute, member_tensor_bytes};
use burn_tensor::{
    Device, DeviceType, Distribution, FloatDType, MemoryAccess, PeakKey, Sharding, Shards, Tensor,
};

/// What `#[compute]` expands to, as a crate depending on `burn` sees it.
mod burn {
    pub use burn_instrument as instrument;
}

/// Rows and columns of the product, and its inner dim before it is split over the members.
const SIDE: usize = 256;

struct Project;

impl Project {
    /// A partial sum, which `exp` cannot take until it is all-reduced.
    #[compute(bounds = Bounds::default(), lead_bounds = Self::lead_bounds)]
    fn forward(&self, lhs: Tensor<2>, rhs: Tensor<2>) -> Tensor<2> {
        lhs.matmul(rhs).exp()
    }

    /// The lead reads its shards of both operands and writes a whole partial sum.
    fn lead_bounds(&self, lhs: &Tensor<2>, rhs: &Tensor<2>) -> Option<Bounds> {
        let [rows, inner] = lhs.shard_dims()?[0];
        let [_, columns] = rhs.shard_dims()?[0];
        Some(Bounds {
            read: member_tensor_bytes(lhs)?[0] + member_tensor_bytes(rhs)?[0],
            write: (rows * columns * size_of::<f32>()) as u64,
            flops: 2 * (rows * inner * columns) as u64,
        })
    }
}

/// Held by each test from its first launch to its last: an instrument's observer is
/// process-wide, and the harness runs tests at once.
static CARDS: Mutex<()> = Mutex::new(());

fn hold_cards() -> MutexGuard<'static, ()> {
    CARDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The runtime's first two cards.
fn cards() -> Vec<Device> {
    #[cfg(feature = "sharded-tests-cuda")]
    let runtime = DeviceType::Cuda;
    #[cfg(all(feature = "sharded-tests-rocm", not(feature = "sharded-tests-cuda")))]
    let runtime = DeviceType::Rocm;
    let cards: Vec<Device> = Device::enumerate(runtime)
        .into_vec()
        .into_iter()
        .take(2)
        .collect();
    assert_eq!(cards.len(), 2, "two cards of the runtime are needed");
    cards
}

/// A sharded compute device of the runtime's first two cards, its members, and the operands of a
/// [`Project`], run once so the communicator is set up and the kernels compiled.
fn two_cards() -> (Device, Vec<Device>, Tensor<2>, Tensor<2>) {
    let members = cards();
    let device = Device::shard_options(&members)
        .init()
        .expect("two cards form a sharded compute device");
    let (lhs, rhs) = operands(&device, &members, &[SIDE / 2; 2]);
    drop(Project.forward(lhs.clone(), rhs.clone()));
    device.sync().expect("the device answers before the run");
    (device, members, lhs, rhs)
}

/// A [`Project`]'s operands on `device`, its inner dim cut into `inner[i]` on member `i`.
fn operands(device: &Device, members: &[Device], inner: &[usize]) -> (Tensor<2>, Tensor<2>) {
    let shards = |shape: fn(usize) -> [usize; 2], dim| {
        let shards = members
            .iter()
            .zip(inner)
            .map(|(member, &inner)| {
                Tensor::<2>::random(shape(inner), Distribution::Default, member)
            })
            .collect();
        Tensor::from_shards(
            Shards {
                shards,
                sharding: Sharding::Split { dim },
            },
            device,
        )
        .expect("the shards split one tensor")
    };
    (
        shards(|inner| [SIDE, inner], 1),
        shards(|inner| [inner, SIDE], 0),
    )
}

#[test]
fn an_all_reduce_is_one_row_with_each_members_part_timed() {
    let _cards = hold_cards();
    let (device, _, lhs, rhs) = two_cards();
    let breakdown =
        Instrument::new(&device).breakdown(|| Project.forward(lhs.clone(), rhs.clone()));

    let per_member = breakdown.per_member.expect("two members are timed apart");
    let rows: Vec<_> = per_member
        .collectives
        .iter()
        .filter(|row| row.operation == "all_reduce")
        .collect();
    let [row] = rows.as_slice() else {
        panic!("one all-reduce row: {:?}", per_member.collectives);
    };
    assert!(
        matches!(row.path, CollectivePath::Native { .. }),
        "{:?}",
        row.path
    );
    assert_eq!(row.bytes, (SIDE * SIDE * 4) as u64);
    assert_eq!(row.members.len(), 2);
    for (index, member) in row.members.iter().enumerate() {
        assert!(!member.part.is_zero(), "member {index}: {row:?}");
    }
}

#[test]
fn each_members_part_of_a_kernel_is_kept() {
    let _cards = hold_cards();
    let (device, _, lhs, rhs) = two_cards();
    let breakdown =
        Instrument::new(&device).breakdown(|| Project.forward(lhs.clone(), rhs.clone()));

    let [project] = breakdown.tree.as_slice() else {
        panic!("one component: {:?}", breakdown.tree);
    };
    let mut each = [Duration::ZERO; 2];
    for kernel in &project.kernels {
        assert_eq!(
            kernel.members.iter().sum::<Duration>(),
            kernel.total,
            "every kernel ran on a member: {kernel:?}"
        );
        for (time, member) in each.iter_mut().zip(&kernel.members) {
            *time += *member;
        }
    }
    assert!(each.iter().all(|time| !time.is_zero()), "{each:?}");
}

#[test]
fn the_device_is_probed_on_its_lead() {
    let _cards = hold_cards();
    let (device, members, _, _) = two_cards();

    let peaks = Peaks::new(&device, FloatDType::F32);
    assert!(peaks.copy_bytes_per_s.is_some(), "{peaks:?}");
    assert_eq!(peaks, Peaks::new(&members[0], FloatDType::F32));
}

/// An uneven split, so the lead's floor is its own larger share and not half the device's.
#[test]
fn the_lead_is_priced_at_its_own_shards() {
    let _cards = hold_cards();
    let (device, members, _, _) = two_cards();
    let inner = [SIDE / 2 + 1, SIDE / 2];
    let (lhs, rhs) = operands(&device, &members, &inner);
    let breakdown =
        Instrument::new(&device).breakdown(|| Project.forward(lhs.clone(), rhs.clone()));

    let [project] = breakdown.tree.as_slice() else {
        panic!("one component: {:?}", breakdown.tree);
    };
    let lead = inner[0];
    assert_eq!(
        project.lead_bounds,
        Some(Bounds {
            read: (2 * SIDE * lead * size_of::<f32>()) as u64,
            write: (SIDE * SIDE * size_of::<f32>()) as u64,
            flops: (2 * SIDE * lead * SIDE) as u64,
        })
    );
}

/// Codex's case: a sharded compute device of one card is a plain device, its lead neither priced
/// nor probed.
#[test]
fn a_sharded_device_of_one_card_is_a_plain_device() {
    let _cards = hold_cards();
    let card = &cards()[..1];
    let device = Device::shard_options(card)
        .init()
        .expect("one card forms a sharded compute device");
    let (lhs, rhs) = operands(&device, card, &[SIDE]);
    let breakdown =
        Instrument::new(&device).breakdown(|| Project.forward(lhs.clone(), rhs.clone()));

    assert_eq!(breakdown.per_member, None);
    let [project] = breakdown.tree.as_slice() else {
        panic!("one component: {:?}", breakdown.tree);
    };
    assert_eq!(project.lead_bounds, None);
    assert_eq!(project.bounds, Some(Bounds::default()));
    let own_copy = (device.measure_peaks(&[PeakKey::Memory(MemoryAccess::Copy)]))
        .into_iter()
        .next()
        .and_then(Result::ok)
        .map(|peak| peak.per_second)
        .filter(|per_second| per_second.is_finite() && *per_second > 0.0);
    assert_eq!(
        Peaks::new(&device, FloatDType::F32).copy_bytes_per_s,
        own_copy
    );
}
