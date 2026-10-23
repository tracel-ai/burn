use burn_backend::ops::CollectiveOps;
use burn_dispatch::Dispatch;
use burn_tensor::{
    DType, Device, TensorData, Tolerance, distributed::ReduceOperation, kind::Basic,
};
use core::ops::Range;

use super::*;

/// One device repeated, which every backend emulates with transfers, slices and concatenations.
/// Four rows over three members split unevenly.
mod emulated {
    use super::*;

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

    #[test]
    fn should_leave_members_held_elsewhere() {
        group().all_reduce_of_held_members();
    }

    #[test]
    #[should_panic(expected = "an empty list of tensors")]
    fn should_refuse_a_group_without_members() {
        TestTensor::<2>::all_reduce(Vec::new(), ReduceOperation::Sum);
    }

    #[test]
    #[should_panic(expected = "different shapes or dtypes")]
    fn should_refuse_members_of_different_shapes() {
        let device = Device::default();
        let members = vec![
            TestTensor::<2>::from_data([[1.0, 2.0]], &device),
            TestTensor::<2>::from_data([[1.0, 2.0, 3.0]], &device),
        ];

        TestTensor::all_gather(members, 0);
    }

    #[test]
    #[should_panic(expected = "a dimension shorter than the group")]
    fn should_refuse_a_dimension_shorter_than_its_members() {
        Group { rows: 2, ..group() }.reduce_scatter();
    }

    /// The split rule belongs to `CollectiveOps`, which asserts it for the callers that reach a
    /// collective without the `Tensor` check: the router's emulation, and a backend itself.
    #[test]
    #[should_panic(expected = "fewer than the 3 members it splits over")]
    fn should_refuse_a_short_dimension_at_the_backend() {
        let group = Group { rows: 2, ..group() };
        let members = group
            .float()
            .into_iter()
            .map(|member| member.into_dispatch())
            .collect();

        Dispatch::float_reduce_scatter(members, 0, ReduceOperation::Sum);
    }

    #[test]
    #[should_panic(expected = "an empty list of devices")]
    fn should_refuse_a_broadcast_to_no_device() {
        let tensor = TestTensor::<2>::from_data([[1.0, 2.0]], &Device::default());

        TestTensor::broadcast(tensor, &[]);
    }

    /// No backend has a quantized collective, and dequantizing to run one would change the
    /// precision of a result the caller asked for in a quantized dtype.
    #[cfg(feature = "quantization")]
    #[test]
    #[should_panic(expected = "Quantized tensors are not supported")]
    fn should_refuse_quantized_members() {
        let device = Device::default();
        let scheme = device.settings().quantization.scheme;
        // A packed store quantizes a last dimension that fills whole u32 words, and no other.
        let plain = TestTensor::<1>::ones([16], &device);
        // Quantized second, where the dtype comparison must not claim a plain mismatch instead.
        let members = vec![plain.clone(), plain.quantize_dynamic(&scheme)];

        TestTensor::all_reduce(members, ReduceOperation::Sum);
    }

    /// `Tensor::cat` drops zero-length members before the backend sees them, which a collective
    /// does not, so a gather over an empty dimension reaches the backend's `cat` with every member
    /// empty.
    #[test]
    fn should_gather_an_empty_dimension() {
        let device = Device::default();
        let members = vec![TestTensor::<2>::zeros([0, 3], &device); 3];

        let gathered = TestTensor::all_gather(members, 0);

        for member in gathered {
            assert_eq!(member.dims(), [0, 3]);
        }
    }

    #[cfg(any(feature = "flex", feature = "cuda"))]
    #[test]
    fn should_average_f16_past_the_range_of_its_sum() {
        group().f16_mean_past_the_range_of_its_sum();
    }

    #[test]
    fn should_move_strided_members() {
        Group {
            strided: true,
            ..group()
        }
        .every_collective();
    }

    #[test]
    fn should_gather_and_scatter_along_a_middle_dim() {
        Group { cols: 6, ..group() }.along_the_middle_dim();
    }

    /// Each member's gradient comes back from its own rows of a gathered output.
    #[test]
    fn should_carry_gradients_through_an_all_gather() {
        let group = group();
        let device = Device::default().autodiff();
        let members: Vec<TestTensor<2>> = (0..group.devices.len())
            .map(|member| {
                TestTensor::<2>::from_data(group.values(member).float(), &device).require_grad()
            })
            .collect();

        let mut outputs = TestTensor::all_gather(members.clone(), 0);
        let gathered = outputs.remove(0);
        let grads = (gathered.clone() * gathered).sum().backward();

        for (member, tensor) in members.iter().enumerate() {
            let expected = group.values(member).multiplied(2).float();
            tensor
                .grad(&grads)
                .expect("every member is gathered")
                .into_data()
                .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
        }
    }

    /// Dispatch runs a backend's collective only when every member is on it. Reductions only: a
    /// bool moved from Flex to a cubecl device fails in `bool_from_data` before any collective.
    #[cfg(feature = "cube")]
    #[test]
    fn should_emulate_members_on_two_backends() {
        let group = Group {
            devices: vec![Device::default(), Device::flex(), Device::default()],
            ..group()
        };
        group.all_reduce();
        group.reduce_scatter();
    }

    fn group() -> Group {
        Group {
            devices: vec![Device::default(); 3],
            rows: 4,
            cols: 3,
            strided: false,
        }
    }
}

/// Every CUDA device, whose collectives run over NCCL when the members allow it.
#[cfg(feature = "distributed")]
mod nccl {
    use super::*;
    use burn_tensor::DeviceType;
    use serial_test::serial;

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
    fn should_emulate_uneven_chunks() {
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

    #[test]
    #[serial]
    fn should_leave_members_held_elsewhere() {
        let Some(group) = group() else { return };
        group.all_reduce_of_held_members();
    }

    #[test]
    #[serial]
    fn should_average_f16_past_the_range_of_its_sum() {
        let Some(group) = group() else { return };
        group.f16_mean_past_the_range_of_its_sum();
    }

    #[test]
    #[serial]
    fn should_move_strided_members() {
        let Some(mut group) = group() else { return };
        group.strided = true;
        group.every_collective();
    }

    #[test]
    #[serial]
    fn should_gather_and_scatter_along_a_middle_dim() {
        let Some(mut group) = group() else { return };
        group.cols *= 2;
        group.along_the_middle_dim();
    }

    /// Even chunks, so every collective runs over NCCL except broadcast and an all-to-all across
    /// two dimensions, which are emulated.
    fn group() -> Option<Group> {
        let devices = Device::enumerate(DeviceType::Cuda).into_vec();
        let members = devices.len();
        (members >= 2).then(|| Group {
            devices,
            rows: 2 * members,
            cols: members,
            strided: false,
        })
    }
}

/// Member `r` holds `rows` by `cols` values counting up from an offset that grows with `r * r`,
/// so sums are negative and no mean divides evenly; laid out column-major when `strided`.
struct Group {
    devices: Vec<Device>,
    rows: usize,
    cols: usize,
    strided: bool,
}

/// Values on the host, row by row: what a member's tensor should hold.
#[derive(Clone)]
struct Grid {
    rows: Vec<Vec<i32>>,
}

impl Group {
    fn every_collective(&self) {
        self.all_reduce();
        self.all_gather();
        self.reduce_scatter();
        self.all_to_all();
        self.broadcast();
    }

    fn all_reduce(&self) {
        let sum = Grid::sum(&self.every_value());
        let members = self.devices.len() as i32;

        let outputs = TestTensor::all_reduce(self.float(), ReduceOperation::Sum);
        self.assert_float(outputs, |_| sum.float());
        let outputs = TestTensor::all_reduce(self.float(), ReduceOperation::Mean);
        self.assert_float(outputs, |_| sum.float_divided(members));

        let outputs = TestTensorInt::all_reduce(self.int(), ReduceOperation::Sum);
        self.assert_values(outputs, |_| sum.int());
        let outputs = TestTensorInt::all_reduce(self.int(), ReduceOperation::Mean);
        self.assert_values(outputs, |_| sum.divided(members).int());
    }

    /// Every member holds 40000, whose sum overflows f16 though the mean does not.
    #[cfg(any(feature = "flex", feature = "cuda"))]
    fn f16_mean_past_the_range_of_its_sum(&self) {
        let members = self
            .devices
            .iter()
            .map(|device| TestTensor::<1>::from_data([40000.0, 40000.0], device).cast(DType::F16));

        let outputs = TestTensor::all_reduce(members.collect(), ReduceOperation::Mean);
        for output in outputs {
            assert_eq!(output.dtype(), DType::F16);
            output.into_data().convert::<f32>().assert_approx_eq::<f32>(
                &TensorData::from([40000.0, 40000.0]),
                Tolerance::default(),
            );
        }
    }

    /// Members still held elsewhere, which an all-reduce in place would overwrite.
    fn all_reduce_of_held_members(&self) {
        let sum = Grid::sum(&self.every_value());
        let held = self.float();

        let outputs = TestTensor::all_reduce(held.clone(), ReduceOperation::Sum);
        self.assert_float(outputs, |_| sum.float());
        self.assert_float(held, |member| self.values(member).float());
    }

    fn all_gather(&self) {
        for dim in [0, 1] {
            let gathered = Grid::cat(self.every_value(), dim);

            let outputs = TestTensor::all_gather(self.float(), dim);
            self.assert_float(outputs, |_| gathered.float());
            let outputs = TestTensorInt::all_gather(self.int(), dim);
            self.assert_values(outputs, |_| gathered.int());
            let outputs = TestTensorBool::all_gather(self.bool(), dim);
            self.assert_values(outputs, |_| gathered.bool());
        }
    }

    fn reduce_scatter(&self) {
        let sum = Grid::sum(&self.every_value());
        let members = self.devices.len() as i32;
        for dim in [0, 1] {
            let chunk = |member| sum.chunk(dim, self.chunk(self.len(dim), member));

            let outputs = TestTensor::reduce_scatter(self.float(), dim, ReduceOperation::Sum);
            self.assert_float(outputs, |member| chunk(member).float());
            let outputs = TestTensor::reduce_scatter(self.float(), dim, ReduceOperation::Mean);
            self.assert_float(outputs, |member| chunk(member).float_divided(members));

            let outputs = TestTensorInt::reduce_scatter(self.int(), dim, ReduceOperation::Sum);
            self.assert_values(outputs, |member| chunk(member).int());
            let outputs = TestTensorInt::reduce_scatter(self.int(), dim, ReduceOperation::Mean);
            self.assert_values(outputs, |member| chunk(member).divided(members).int());
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

            let outputs = TestTensor::all_to_all(self.float(), split_dim, concat_dim);
            self.assert_float(outputs, |member| received(member).float());
            let outputs = TestTensorInt::all_to_all(self.int(), split_dim, concat_dim);
            self.assert_values(outputs, |member| received(member).int());
            let outputs = TestTensorBool::all_to_all(self.bool(), split_dim, concat_dim);
            self.assert_values(outputs, |member| received(member).bool());
        }
    }

    /// Broadcasts from the last member, so the root is not the first destination.
    fn broadcast(&self) {
        let root = self.devices.len() - 1;
        let values = self.values(root);

        let outputs = TestTensor::broadcast(self.float().remove(root), &self.devices);
        self.assert_float(outputs, |_| values.float());
        let outputs = TestTensorInt::broadcast(self.int().remove(root), &self.devices);
        self.assert_values(outputs, |_| values.int());
        let outputs = TestTensorBool::broadcast(self.bool().remove(root), &self.devices);
        self.assert_values(outputs, |_| values.bool());
    }

    /// Each member's values as `[rows, cols / 2, 2]`, gathered and scattered along the middle
    /// dimension. A middle index spans two columns, so the expectations are over the columns.
    fn along_the_middle_dim(&self) {
        let middle = self.cols / 2;
        let gathered = Grid::cat(self.every_value(), 1);
        let sum = Grid::sum(&self.every_value());
        let chunk = |member| {
            let range = self.chunk(middle, member);
            sum.chunk(1, 2 * range.start..2 * range.end)
        };

        let outputs = TestTensor::all_gather(self.float_3d(), 1);
        self.assert_float(Self::flattened(outputs), |_| gathered.float());
        let outputs = TestTensor::reduce_scatter(self.float_3d(), 1, ReduceOperation::Sum);
        self.assert_float(Self::flattened(outputs), |member| chunk(member).float());
    }

    fn values(&self, member: usize) -> Grid {
        let len = self.rows * self.cols;
        let first = (member * len + member * member) as i32 - (len * self.devices.len()) as i32;
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

    fn float(&self) -> Vec<TestTensor<2>> {
        self.members(|values, device| match self.strided {
            true => TestTensor::<2>::from_data(values.transposed().float(), device).transpose(),
            false => TestTensor::<2>::from_data(values.float(), device),
        })
    }

    fn int(&self) -> Vec<TestTensorInt<2>> {
        self.members(|values, device| match self.strided {
            true => TestTensorInt::<2>::from_data(values.transposed().int(), device).transpose(),
            false => TestTensorInt::<2>::from_data(values.int(), device),
        })
    }

    fn bool(&self) -> Vec<TestTensorBool<2>> {
        self.members(|values, device| match self.strided {
            true => TestTensorBool::<2>::from_data(values.transposed().bool(), device).transpose(),
            false => TestTensorBool::<2>::from_data(values.bool(), device),
        })
    }

    fn float_3d(&self) -> Vec<TestTensor<3>> {
        self.members(|values, device| {
            TestTensor::<2>::from_data(values.float(), device).reshape([
                self.rows,
                self.cols / 2,
                2,
            ])
        })
    }

    fn members<const D: usize, K: Basic>(
        &self,
        tensor: impl Fn(Grid, &Device) -> Tensor<D, K>,
    ) -> Vec<Tensor<D, K>> {
        self.devices
            .iter()
            .enumerate()
            .map(|(member, device)| tensor(self.values(member), device))
            .collect()
    }

    fn flattened(outputs: Vec<TestTensor<3>>) -> Vec<TestTensor<2>> {
        outputs
            .into_iter()
            .map(|output| {
                let [rows, middle, pair] = output.dims();
                output.reshape([rows, middle * pair])
            })
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

    fn assert_float(&self, outputs: Vec<TestTensor<2>>, expected: impl Fn(usize) -> TensorData) {
        self.assert_members(outputs.len());
        for (member, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.device(), self.devices[member], "member {member}");
            output
                .into_data()
                .assert_approx_eq::<FloatElem>(&expected(member), Tolerance::default());
        }
    }

    fn assert_values<K: Basic>(
        &self,
        outputs: Vec<Tensor<2, K>>,
        expected: impl Fn(usize) -> TensorData,
    ) {
        self.assert_members(outputs.len());
        for (member, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.device(), self.devices[member], "member {member}");
            output.into_data().assert_eq(&expected(member), false);
        }
    }

    fn assert_members(&self, outputs: usize) {
        assert_eq!(outputs, self.devices.len(), "one output per member");
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

    fn transposed(&self) -> Grid {
        let rows = (0..self.rows[0].len())
            .map(|col| self.rows.iter().map(|row| row[col]).collect())
            .collect();
        Grid { rows }
    }

    fn multiplied(&self, factor: i32) -> Grid {
        self.mapped(|value| value * factor)
    }

    fn divided(&self, divisor: i32) -> Grid {
        self.mapped(|value| value / divisor)
    }

    fn mapped(&self, value: impl Fn(i32) -> i32) -> Grid {
        let rows = self
            .rows
            .iter()
            .map(|row| row.iter().map(|it| value(*it)).collect())
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
