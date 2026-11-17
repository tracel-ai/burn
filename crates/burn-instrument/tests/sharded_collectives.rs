//! What an instrumented run on a sharded compute device of two cards records of its collectives:
//! one row per kind, each member's part timed on its own card.
#![cfg(any(feature = "sharded-tests-cuda", feature = "sharded-tests-rocm"))]

use burn_instrument::{Bounds, CollectivePath, Instrument, compute};
use burn_tensor::{Device, DeviceType, Distribution, Sharding, Shards, Tensor};

/// What `#[compute]` expands to, as a crate depending on `burn` sees it.
mod burn {
    pub use burn_instrument as instrument;
}

/// Rows and columns of the product, and its inner dim before it is split over the members.
const SIDE: usize = 256;

struct Project;

impl Project {
    /// A partial sum, which `exp` cannot take until it is all-reduced.
    #[compute(bounds = Bounds::default())]
    fn forward(&self, lhs: Tensor<2>, rhs: Tensor<2>) -> Tensor<2> {
        lhs.matmul(rhs).exp()
    }
}

#[test]
fn an_all_reduce_is_one_row_with_each_members_part_timed() {
    #[cfg(feature = "sharded-tests-cuda")]
    let runtime = DeviceType::Cuda;
    #[cfg(all(feature = "sharded-tests-rocm", not(feature = "sharded-tests-cuda")))]
    let runtime = DeviceType::Rocm;
    let members: Vec<Device> = Device::enumerate(runtime)
        .into_vec()
        .into_iter()
        .take(2)
        .collect();
    assert_eq!(members.len(), 2, "two cards of the runtime are needed");
    let device = Device::shard_options(&members)
        .init()
        .expect("two cards form a sharded compute device");
    let shards = |shape: [usize; 2], dim| {
        let shards = members
            .iter()
            .map(|member| Tensor::<2>::random(shape, Distribution::Default, member))
            .collect();
        Tensor::from_shards(
            Shards {
                shards,
                sharding: Sharding::Split { dim },
            },
            &device,
        )
        .expect("the shards split one tensor evenly")
    };
    let lhs = shards([SIDE, SIDE / 2], 1);
    let rhs = shards([SIDE / 2, SIDE], 0);
    // Once before the run, so the communicator is set up and the kernels compiled.
    drop(Project.forward(lhs.clone(), rhs.clone()));
    device.sync().expect("the device answers before the run");

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
