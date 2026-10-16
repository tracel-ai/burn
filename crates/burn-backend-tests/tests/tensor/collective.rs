use burn_backend::{distributed::ReduceOperation, ops::CollectiveOps};
use burn_dispatch::{Dispatch, DispatchDevice, DispatchTensor};
use burn_tensor::{Device, TensorData, Tolerance};
use core::ops::Range;

use super::*;

/// Member `r` holds `rows` by `cols` values counting up from `r * rows * cols`.
struct Group {
    devices: Vec<Device>,
    rows: usize,
    cols: usize,
}

/// Values on the host, row by row: what a member's tensor should hold.
#[derive(Clone)]
struct Grid {
    rows: Vec<Vec<i32>>,
}

impl Group {
    fn values(&self, member: usize) -> Grid {
        let first = (member * self.rows * self.cols) as i32;
        let rows = (0..self.rows)
            .map(|row| {
                let start = first + (row * self.cols) as i32;
                (start..start + self.cols as i32).collect()
            })
            .collect();
        Grid { rows }
    }

    fn every_value(&self) -> Vec<Grid> {
        (0..self.devices.len())
            .map(|member| self.values(member))
            .collect()
    }

    fn float(&self) -> Vec<DispatchTensor> {
        self.devices
            .iter()
            .enumerate()
            .map(|(member, device)| {
                TestTensor::<2>::from_data(self.values(member).float(), device).into_dispatch()
            })
            .collect()
    }

    fn int(&self) -> Vec<DispatchTensor> {
        self.devices
            .iter()
            .enumerate()
            .map(|(member, device)| {
                TestTensorInt::<2>::from_data(self.values(member).int(), device).into_dispatch()
            })
            .collect()
    }

    fn bool(&self) -> Vec<DispatchTensor> {
        self.devices
            .iter()
            .enumerate()
            .map(|(member, device)| {
                TestTensorBool::<2>::from_data(self.values(member).bool(), device).into_dispatch()
            })
            .collect()
    }

    fn dispatch_devices(&self) -> Vec<DispatchDevice> {
        self.devices
            .iter()
            .map(|device| device.as_dispatch().clone())
            .collect()
    }

    fn chunk(&self, len: usize, member: usize) -> Range<usize> {
        let members = self.devices.len();
        let (base, longer) = (len / members, len % members);
        let start = member * base + member.min(longer);
        start..start + base + usize::from(member < longer)
    }

    fn len(&self, dim: usize) -> usize {
        [self.rows, self.cols][dim]
    }

    fn assert_float(&self, outputs: Vec<DispatchTensor>, expected: impl Fn(usize) -> TensorData) {
        self.assert_members(outputs.len());
        for (member, output) in outputs.into_iter().enumerate() {
            let output = TestTensor::<2>::from_dispatch(output);
            assert_eq!(output.device(), self.devices[member], "member {member}");
            output
                .into_data()
                .assert_approx_eq::<FloatElem>(&expected(member), Tolerance::default());
        }
    }

    fn assert_int(&self, outputs: Vec<DispatchTensor>, expected: impl Fn(usize) -> TensorData) {
        self.assert_members(outputs.len());
        for (member, output) in outputs.into_iter().enumerate() {
            let output = TestTensorInt::<2>::from_dispatch(output);
            assert_eq!(output.device(), self.devices[member], "member {member}");
            output.into_data().assert_eq(&expected(member), false);
        }
    }

    fn assert_bool(&self, outputs: Vec<DispatchTensor>, expected: impl Fn(usize) -> TensorData) {
        self.assert_members(outputs.len());
        for (member, output) in outputs.into_iter().enumerate() {
            let output = TestTensorBool::<2>::from_dispatch(output);
            assert_eq!(output.device(), self.devices[member], "member {member}");
            output.into_data().assert_eq(&expected(member), false);
        }
    }

    fn assert_members(&self, outputs: usize) {
        assert_eq!(outputs, self.devices.len(), "one output per member");
    }

    fn all_reduce(&self) {
        let sum = Grid::sum(&self.every_value());
        let members = self.devices.len() as i32;

        let outputs = Dispatch::float_all_reduce(self.float(), ReduceOperation::Sum);
        self.assert_float(outputs, |_| sum.float());
        let outputs = Dispatch::float_all_reduce(self.float(), ReduceOperation::Mean);
        self.assert_float(outputs, |_| sum.float_divided(members));

        let outputs = Dispatch::int_all_reduce(self.int(), ReduceOperation::Sum);
        self.assert_int(outputs, |_| sum.int());
        let outputs = Dispatch::int_all_reduce(self.int(), ReduceOperation::Mean);
        self.assert_int(outputs, |_| sum.divided(members).int());
    }

    fn all_gather(&self) {
        for dim in [0, 1] {
            let gathered = Grid::cat(self.every_value(), dim);

            let outputs = Dispatch::float_all_gather(self.float(), dim);
            self.assert_float(outputs, |_| gathered.float());
            let outputs = Dispatch::int_all_gather(self.int(), dim);
            self.assert_int(outputs, |_| gathered.int());
            let outputs = Dispatch::bool_all_gather(self.bool(), dim);
            self.assert_bool(outputs, |_| gathered.bool());
        }
    }

    fn reduce_scatter(&self) {
        let sum = Grid::sum(&self.every_value());
        for dim in [0, 1] {
            let chunk = |member| sum.chunk(dim, self.chunk(self.len(dim), member));
            let op = ReduceOperation::Sum;

            let outputs = Dispatch::float_reduce_scatter(self.float(), dim, op);
            self.assert_float(outputs, |member| chunk(member).float());
            let outputs = Dispatch::int_reduce_scatter(self.int(), dim, op);
            self.assert_int(outputs, |member| chunk(member).int());
        }
    }

    fn all_to_all(&self) {
        for [split_dim, concat_dim] in [[0, 0], [1, 1], [0, 1]] {
            let received = |member| {
                let range = self.chunk(self.len(split_dim), member);
                let chunks = self
                    .every_value()
                    .iter()
                    .map(|values| values.chunk(split_dim, range.clone()))
                    .collect();
                Grid::cat(chunks, concat_dim)
            };

            let outputs = Dispatch::float_all_to_all(self.float(), split_dim, concat_dim);
            self.assert_float(outputs, |member| received(member).float());
            let outputs = Dispatch::int_all_to_all(self.int(), split_dim, concat_dim);
            self.assert_int(outputs, |member| received(member).int());
            let outputs = Dispatch::bool_all_to_all(self.bool(), split_dim, concat_dim);
            self.assert_bool(outputs, |member| received(member).bool());
        }
    }

    /// Broadcasts from the last member, so the root is not the first destination.
    fn broadcast(&self) {
        let root = self.devices.len() - 1;
        let values = self.values(root);
        let devices = self.dispatch_devices();

        let outputs = Dispatch::float_broadcast(self.float().remove(root), &devices);
        self.assert_float(outputs, |_| values.float());
        let outputs = Dispatch::int_broadcast(self.int().remove(root), &devices);
        self.assert_int(outputs, |_| values.int());
        let outputs = Dispatch::bool_broadcast(self.bool().remove(root), &devices);
        self.assert_bool(outputs, |_| values.bool());
    }
}

impl Grid {
    fn sum(grids: &[Grid]) -> Grid {
        let mut sum = grids[0].clone();
        for grid in &grids[1..] {
            for (sum, row) in sum.rows.iter_mut().zip(&grid.rows) {
                for (sum, value) in sum.iter_mut().zip(row) {
                    *sum += value;
                }
            }
        }
        sum
    }

    fn cat(grids: Vec<Grid>, dim: usize) -> Grid {
        let mut grids = grids.into_iter();
        let mut cat = grids.next().expect("a group has a member");
        for grid in grids {
            match dim {
                0 => cat.rows.extend(grid.rows),
                _ => {
                    for (row, more) in cat.rows.iter_mut().zip(grid.rows) {
                        row.extend(more);
                    }
                }
            }
        }
        cat
    }

    fn chunk(&self, dim: usize, range: Range<usize>) -> Grid {
        let rows = match dim {
            0 => self.rows[range].to_vec(),
            _ => self
                .rows
                .iter()
                .map(|row| row[range.clone()].to_vec())
                .collect(),
        };
        Grid { rows }
    }

    fn divided(&self, divisor: i32) -> Grid {
        let rows = self
            .rows
            .iter()
            .map(|row| row.iter().map(|value| value / divisor).collect())
            .collect();
        Grid { rows }
    }

    fn float_divided(&self, divisor: i32) -> TensorData {
        let values = self.flat().map(|value| value as f32 / divisor as f32);
        TensorData::new(values.collect(), self.shape())
    }

    fn flat(&self) -> impl Iterator<Item = i32> + '_ {
        self.rows.iter().flatten().copied()
    }

    fn shape(&self) -> [usize; 2] {
        [self.rows.len(), self.rows[0].len()]
    }

    fn float(&self) -> TensorData {
        TensorData::new(
            self.flat().map(|value| value as f32).collect(),
            self.shape(),
        )
    }

    fn int(&self) -> TensorData {
        TensorData::new(self.flat().collect(), self.shape())
    }

    fn bool(&self) -> TensorData {
        TensorData::new(
            self.flat().map(|value| value % 3 == 0).collect(),
            self.shape(),
        )
    }
}

/// One device repeated, which every backend composes from transfers, slices and concatenations.
/// Four rows over three members split unevenly.
mod composed {
    use super::*;

    fn group() -> Group {
        Group {
            devices: vec![Device::default(); 3],
            rows: 4,
            cols: 3,
        }
    }

    #[test]
    fn should_all_reduce() {
        group().all_reduce();
    }

    #[test]
    fn should_all_gather() {
        group().all_gather();
    }

    #[test]
    fn should_reduce_scatter() {
        group().reduce_scatter();
    }

    #[test]
    fn should_all_to_all() {
        group().all_to_all();
    }

    #[test]
    fn should_broadcast() {
        group().broadcast();
    }
}

/// Every CUDA device, whose collectives run over NCCL when the members allow it.
#[cfg(feature = "distributed")]
mod nccl {
    use super::*;
    use burn_tensor::DeviceType;
    use serial_test::serial;

    /// Every chunk even, so each collective runs over NCCL.
    fn group() -> Option<Group> {
        let devices = Device::enumerate(DeviceType::Cuda).into_vec();
        let members = devices.len();
        (members >= 2).then(|| Group {
            devices,
            rows: 2 * members,
            cols: members,
        })
    }

    #[test]
    #[serial]
    fn should_all_reduce() {
        let Some(group) = group() else { return };
        group.all_reduce();
    }

    #[test]
    #[serial]
    fn should_all_gather() {
        let Some(group) = group() else { return };
        group.all_gather();
    }

    #[test]
    #[serial]
    fn should_reduce_scatter() {
        let Some(group) = group() else { return };
        group.reduce_scatter();
    }

    #[test]
    #[serial]
    fn should_all_to_all() {
        let Some(group) = group() else { return };
        group.all_to_all();
    }

    #[test]
    #[serial]
    fn should_broadcast() {
        let Some(group) = group() else { return };
        group.broadcast();
    }

    #[test]
    #[serial]
    fn should_compose_uneven_chunks() {
        let Some(mut group) = group() else { return };
        group.rows += 1;
        group.cols += 1;
        group.reduce_scatter();
        group.all_to_all();
    }

    #[test]
    #[serial]
    fn should_keep_member_order_over_descending_devices() {
        let Some(mut group) = group() else { return };
        group.devices.reverse();
        group.all_reduce();
        group.all_gather();
        group.reduce_scatter();
        group.all_to_all();
    }

    /// One row of three values, which a pitched allocation pads past its elements.
    #[test]
    #[serial]
    fn should_move_single_row_members() {
        let Some(mut group) = group() else { return };
        group.rows = 1;
        group.cols = 3;
        group.all_reduce();
        group.all_gather();
        group.broadcast();
    }
}
