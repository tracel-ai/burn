use crate::engine::{
    codegen::ir::{FuseArg, FuseOp, FuseType},
    trace::block::FuseBlock,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_ir::{TensorId, TensorIr};
use burn_std::{DType, Shape, Strides};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

#[cfg(feature = "autotune-checks")]
use crate::CubeFusionHandle;
#[cfg(feature = "autotune-checks")]
use burn_backend::TensorData;
#[cfg(feature = "autotune-checks")]
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize, Debug)]
/// A trace contains all [blocks](super::block::FuseBlock) and the [resources](FuseResources) used by the
/// kernel.
pub struct FuseTrace {
    pub blocks: Vec<FuseBlock>,
    pub resources: FuseResources,
}

impl FuseTrace {
    /// The highest relative shape id any of this trace's blocks names as its reference shape.
    pub fn max_relative_shape_id(&self) -> Option<usize> {
        self.blocks
            .iter()
            .flat_map(|block| block.shape_ref.iter().copied())
            .max()
    }

    /// Bytes a kernel running this trace moves at least, with the relative ids resolved against
    /// the context's `tensors`: every input read once, every output written once, and every
    /// output a later block reads read back once. An intermediate the trace keeps in registers is
    /// in none of them.
    pub fn traffic(&self, tensors: &hashbrown::HashMap<TensorId, TensorIr>) -> usize {
        let resolve = |registered: &RegisterTensor| match registered {
            RegisterTensor::Normal(tensor, _) | RegisterTensor::QuantValues(tensor) => {
                tensors.get(&tensor.id)
            }
            RegisterTensor::QuantParams(_) => None,
        };
        let indexed = self.indexed_reads(tensors);

        let inputs: usize = self
            .resources
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(pos, input)| {
                let tensor = resolve(input)?;
                let elems = indexed.get(&pos).copied();
                Some(bytes(tensor, elems.unwrap_or(tensor.shape.num_elements())))
            })
            .sum();
        let outputs_and_read_backs: usize = self
            .resources
            .outputs
            .iter()
            .chain(self.resources.buffers.iter())
            .filter_map(resolve)
            .map(|tensor| bytes(tensor, tensor.shape.num_elements()))
            .sum();

        inputs + outputs_and_read_backs
    }

    /// Elements read at least from each input of an indexing op, by its position in the inputs.
    ///
    /// Such an input is read by indexing ops alone. A concatenation reads each of its inputs
    /// whole, and a gather or a select reads its indices whole. Which elements of its source
    /// they read depends on the index values, which only the device holds, so the source is
    /// charged as if every index were equal: one element per position outside the indexed dim.
    /// An input several ops read is charged the most any of them reads.
    fn indexed_reads(
        &self,
        tensors: &hashbrown::HashMap<TensorId, TensorIr>,
    ) -> BTreeMap<usize, usize> {
        let shape = |arg: &FuseArg| {
            match arg {
                FuseArg::Input(pos, ..) => self.resources.inputs.get_id(*pos),
                _ => None,
            }
            .and_then(|id| tensors.get(&id))
            .map(|tensor| &tensor.shape)
        };
        let whole = |arg: &FuseArg| shape(arg).map_or(0, |shape| shape.num_elements());
        let outside = |arg: &FuseArg, dim: usize| {
            shape(arg).map_or(0, |shape| shape.num_elements() / shape[dim].max(1))
        };

        let mut reads = BTreeMap::new();
        let mut read = |arg: &FuseArg, elems: usize| {
            if let FuseArg::Input(pos, ..) = arg {
                let most = reads.entry(*pos).or_insert(0);
                *most = usize::max(*most, elems);
            }
        };

        for op in self.blocks.iter().flat_map(|block| &block.ops) {
            match op {
                // One source element per index, so the positions outside `dim` are the indices'.
                FuseOp::Gather {
                    input,
                    indices,
                    dim,
                    ..
                } => {
                    read(input, outside(indices, *dim));
                    read(indices, whole(indices));
                }
                FuseOp::Select {
                    input,
                    indices,
                    dim,
                    ..
                } => {
                    // Every index equal reads one slice, and no index reads none.
                    let slices = whole(indices).min(1);
                    read(input, slices * outside(input, *dim));
                    read(indices, whole(indices));
                }
                FuseOp::Cat { inputs, .. } => {
                    for input in inputs {
                        read(input, whole(input));
                    }
                }
                _ => {}
            }
        }

        reads
    }
}

/// Bytes `elems` elements of `tensor` take in global memory.
fn bytes(tensor: &TensorIr, elems: usize) -> usize {
    match tensor.dtype {
        // Packed values move at their quantized width, not the width of the word they sit in.
        DType::QFloat(scheme) => (elems * scheme.size_bits_value()).div_ceil(8),
        dtype => elems * dtype_to_storage_type(dtype).size(),
    }
}

impl core::fmt::Display for FuseTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FuseTrace")?;
        for b in self.blocks.iter() {
            writeln!(f, " - Block shape={:?}", b.shape_ref)?;
            for (tensor, ops) in b.reads.iter() {
                for op in ops.iter() {
                    writeln!(f, "   - {op} <== {tensor}")?;
                }
            }
            for op in b.ops.iter() {
                writeln!(f, "   - {op}")?;
            }
            for (tensor, ops) in b.writes.iter() {
                for op in ops.iter() {
                    writeln!(f, "   - {op} <== {tensor}")?;
                }
            }
        }

        Ok(())
    }
}

pub enum TuneOutput {
    UnChecked,
    #[cfg(feature = "autotune-checks")]
    Checked {
        handles: HashMap<TensorId, (Shape, CubeFusionHandle)>,
    },
}

impl TuneOutput {
    #[allow(unused_variables)]
    pub fn merge(self, other: Self) -> Self {
        let mut result = self;

        match &mut result {
            TuneOutput::UnChecked => {}
            #[cfg(feature = "autotune-checks")]
            TuneOutput::Checked { handles } => match other {
                TuneOutput::UnChecked => {}
                TuneOutput::Checked { handles: o } => {
                    for (k, v) in o.into_iter() {
                        handles.insert(k, v);
                    }
                }
            },
        }

        result
    }
}

impl cubecl::tune::AutotuneOutput for TuneOutput {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        use burn_backend::Tolerance;
        use burn_std::DType;

        if let (
            TuneOutput::Checked {
                handles: handles_ref,
            },
            TuneOutput::Checked { handles },
        ) = (self, &other)
        {
            let mut num_checked = 0;
            let mut num_handles = 0;
            for (id, (shape, handle)) in handles_ref.iter() {
                num_handles += 1;
                if let Some((shape_other, other)) = handles.get(id) {
                    use burn_std::is_contiguous;
                    use cubecl::std::tensor::into_contiguous;

                    let current_handle = if !is_contiguous(shape, &handle.strides) {
                        into_contiguous(
                            &handle.client,
                            handle.clone().binding(shape.clone()),
                            dtype_to_storage_type(handle.dtype),
                        )
                        .handle
                    } else {
                        handle.handle.clone()
                    };
                    let other_handle = if !is_contiguous(shape, &other.strides) {
                        into_contiguous(
                            &other.client,
                            other.clone().binding(shape.clone()),
                            dtype_to_storage_type(other.dtype),
                        )
                        .handle
                    } else {
                        other.handle.clone()
                    };

                    let data_ref = handle.client.read_one(current_handle).unwrap();
                    let data_other = other.client.read_one(other_handle).unwrap();
                    let data_ref = TensorData::from_bytes(data_ref, shape.clone(), handle.dtype);
                    let data_other =
                        TensorData::from_bytes(data_other, shape_other.clone(), handle.dtype);

                    match handle.dtype {
                        DType::F64 => {
                            data_ref.assert_approx_eq::<f64>(&data_other, Tolerance::permissive())
                        }
                        DType::F32 => {
                            data_ref.assert_approx_eq::<f32>(&data_other, Tolerance::permissive())
                        }
                        DType::F16 => data_ref
                            .assert_approx_eq::<half::f16>(&data_other, Tolerance::permissive()),
                        DType::BF16 => data_ref
                            .assert_approx_eq::<half::bf16>(&data_other, Tolerance::permissive()),
                        _ => data_ref.assert_eq(&data_other, true),
                    }
                    num_checked += 1;
                } else {
                    // Debug info for the tests.
                    println!("No tensor found for {id:?}=>{shape:?}");
                }
            }

            // At least one check is needed per output when there is an output.
            //
            // Some optimizations might write more outputs than needed, so it might be fined if
            // the number of handles is different, but at least one is required.
            //
            // An optimization might not create outputs if its dead code detection is triggered,
            // therefore avoiding useless computation.
            if num_handles > 0 {
                assert!(num_checked >= 1);
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
/// Declare all resources used by the kernel, and potentially multiple [blocks](super::block::FuseBlock).
///
/// # Notes
///
/// Each block can't contain their own resources, since they are shared between blocks. The
/// vectorization factor of one input tensor must be the same for all blocks.
pub struct FuseResources {
    pub outputs: RegisteredTensors,
    pub inputs: RegisteredTensors,
    pub scalars: Vec<(FuseType, u64)>,
    // TODO: Making put a map of global registers.
    pub views: Vec<TensorView>,
    pub indexed: BTreeMap<TensorId, FuseArg>,
    pub inputs_unhandled: Vec<TensorId>,
    pub outputs_unhandled: Vec<FuseArg>,
    pub num_reshaped: usize,
    /// Necessary to remove some entries from the context.
    pub dropped: HashSet<TensorId>,
    /// We know during fusion that we have to have those buffers has global.
    /// The pos here can be interpreted as GLOBAL pos where the output pos are locals.
    pub buffers: RegisteredTensors,
    /// Global registers available everywhere.
    ///
    /// TODO: Not all registers should be globals.
    pub registers: BTreeMap<TensorId, FuseArg>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct RuntimeLayout {
    pub shape: Shape,
    pub strides: Strides,
}

impl Default for RuntimeLayout {
    fn default() -> Self {
        Self {
            shape: Shape::new([]),
            strides: Strides::new(&[]),
        }
    }
}

#[derive(Debug)]
pub enum TraceError<Err> {
    ReferenceNotFound,
    RunnerError(Err),
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum TensorView {
    Reshape {
        reshaped: TensorId,
        original: TensorId,
        reshape_pos: usize,
        shape_relative: Shape,
    },
    SwapDims {
        swapped: TensorId,
        original: TensorId,
        dims: (usize, usize),
    },
    NhwcStrides {
        id: TensorId,
        stride_relayout: Shape,
    },
}

#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub struct RegisteredTensors {
    tensors: Vec<RegisterTensor>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum RegisterTensor {
    Normal(TensorIr, FuseType),
    QuantValues(TensorIr),
    QuantParams(TensorId),
}

impl RegisterTensor {
    pub fn as_normal_tensor(&self) -> Option<(&TensorIr, &FuseType)> {
        match self {
            RegisterTensor::Normal(tensor_ir, precision) => Some((tensor_ir, precision)),
            RegisterTensor::QuantValues(_) => None,
            RegisterTensor::QuantParams(_) => None,
        }
    }
}

impl IntoIterator for RegisteredTensors {
    type Item = RegisterTensor;
    type IntoIter = std::vec::IntoIter<RegisterTensor>;

    /// Consumes and iterate over all the registered tensors.
    fn into_iter(self) -> Self::IntoIter {
        self.tensors.into_iter()
    }
}

impl RegisteredTensors {
    /// Iterate over all the registered tensors.
    pub fn iter(&self) -> impl Iterator<Item = &RegisterTensor> {
        self.tensors.iter()
    }

    /// Returns the number of tensors registered.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns whether no tensor is registered.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Retrieve the [tensor id](TensorId) at the given index.
    pub fn get_id(&self, index: usize) -> Option<TensorId> {
        self.tensors.get(index).map(|entry| match entry {
            RegisterTensor::Normal(tensor_ir, _) => tensor_ir.id,
            RegisterTensor::QuantValues(tensor_ir) => tensor_ir.id,
            RegisterTensor::QuantParams(tensor_id) => *tensor_id,
        })
    }

    /// Doesn't return quantized tensor.
    pub fn get_index(&self, tensor_id: TensorId) -> Option<usize> {
        self.tensors
            .iter()
            .enumerate()
            .find(|(_pos, entry)| match entry {
                RegisterTensor::Normal(tensor_ir, _) => tensor_ir.id == tensor_id,
                RegisterTensor::QuantValues(_) => false,
                RegisterTensor::QuantParams(_) => false,
            })
            .map(|(pos, _)| pos)
    }

    /// Get the index of a quantized tensor.
    pub fn get_index_quant(&self, tensor_id: TensorId) -> Option<usize> {
        self.tensors
            .iter()
            .enumerate()
            .find(|(_pos, entry)| match entry {
                RegisterTensor::Normal(..) => false,
                RegisterTensor::QuantValues(tensor_ir) => tensor_ir.id == tensor_id,
                RegisterTensor::QuantParams(_) => false,
            })
            .map(|(pos, _)| pos)
    }

    /// Doesn't return quantized tensor.
    pub fn get(&self, tensor_id: TensorId) -> Option<(&TensorIr, &FuseType)> {
        self.tensors
            .iter()
            .find(|entry| match entry {
                RegisterTensor::Normal(tensor_ir, _) => tensor_ir.id == tensor_id,
                RegisterTensor::QuantValues(_) => false,
                RegisterTensor::QuantParams(_) => false,
            })
            .and_then(|entry| match entry {
                RegisterTensor::Normal(tensor_ir, fuse_precision) => {
                    Some((tensor_ir, fuse_precision))
                }
                RegisterTensor::QuantValues(_) => None,
                RegisterTensor::QuantParams(_) => None,
            })
    }

    /// Insert a quantized tensor.
    ///
    /// It will return the positions for both the value tensor and param tensor.
    pub fn insert_quant(&mut self, tensor: TensorIr) -> (usize, usize) {
        if let Some(old) = self.tensors.iter().enumerate().find(|(_, val)| match &val {
            RegisterTensor::QuantValues(tensor_ir) => tensor_ir == &tensor,
            _ => false,
        }) {
            let values = old.0;
            let params = values + 1;
            return (values, params);
        }

        let params = RegisterTensor::QuantParams(tensor.id);
        let values = RegisterTensor::QuantValues(tensor);
        let pos_values = self.len();
        self.tensors.push(values);

        let pos_params = self.len();
        self.tensors.push(params);

        (pos_values, pos_params)
    }

    /// Insert a normal tensor with the given [type](FuseType) in the current block.
    pub fn insert(&mut self, precision: FuseType, tensor: TensorIr) -> usize {
        if let Some(old) = self.tensors.iter().enumerate().find(|(_, val)| match &val {
            RegisterTensor::Normal(tensor_ir, _) => tensor_ir.id == tensor.id,
            _ => false,
        }) {
            return old.0;
        }

        let value = RegisterTensor::Normal(tensor, precision);
        let pos = self.len();

        self.tensors.push(value);

        pos
    }

    /// Update the already registered tensor with the given [tensor ir](TensorIr).
    ///
    /// # Notes
    ///
    /// This function only works with normal tensors, not quantized tensors.
    pub fn update(&mut self, tensor: &TensorIr) {
        if let Some(entry) = self.tensors.iter_mut().find(|entry| match entry {
            RegisterTensor::Normal(tensor_ir, _) => tensor_ir.id == tensor.id,
            _ => false,
        }) && let RegisterTensor::Normal(tensor_ir, _) = entry
        {
            tensor_ir.status = tensor.status
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{settings::FuseSettings, trace::TraceFuser};
    use burn_ir::TensorStatus;

    const COLS: usize = 64;
    const F32: usize = 4;
    const I32: usize = 4;

    fn tensor(id: u64, dims: &[usize], dtype: DType) -> TensorIr {
        TensorIr {
            id: TensorId::new(id),
            shape: Shape::from(dims.to_vec()),
            status: TensorStatus::ReadOnly,
            dtype,
        }
    }

    /// The traffic of the trace `fuse` builds over a `[rows, COLS]` table, `fuse` returning the
    /// tensors other than the table it registered.
    fn traffic(rows: usize, fuse: fn(&mut TraceFuser, &TensorIr) -> Vec<TensorIr>) -> usize {
        let table = tensor(0, &[rows, COLS], DType::F32);
        let mut fuser = TraceFuser::new(FuseSettings::default());
        let mut registered = fuse(&mut fuser, &table);
        let trace = fuser.finish(Shape::new([4, COLS]));

        registered.push(table);
        let tensors = registered.into_iter().map(|tensor| (tensor.id, tensor));
        trace.traffic(&tensors.collect())
    }

    /// Selects 4 rows of the table.
    fn select(fuser: &mut TraceFuser, table: &TensorIr) -> Vec<TensorIr> {
        select_rows(fuser, table, 4)
    }

    /// Selects no row of the table.
    fn select_none(fuser: &mut TraceFuser, table: &TensorIr) -> Vec<TensorIr> {
        select_rows(fuser, table, 0)
    }

    fn select_rows(fuser: &mut TraceFuser, table: &TensorIr, rows: usize) -> Vec<TensorIr> {
        let indices = tensor(1, &[rows], DType::I32);
        let out = tensor(2, &[rows, COLS], DType::F32);
        let op = FuseOp::Select {
            input: fuser.input_indexed(table).unwrap(),
            indices: fuser.input_indexed(&indices).unwrap(),
            output: fuser.output(&out).unwrap(),
            dim: 0,
        };
        fuser.fuse_operation(op);
        vec![indices, out]
    }

    /// Gathers 4 elements of each column of the table.
    fn gather(fuser: &mut TraceFuser, table: &TensorIr) -> Vec<TensorIr> {
        let indices = tensor(1, &[4, COLS], DType::I32);
        let out = tensor(2, &[4, COLS], DType::F32);
        let op = FuseOp::Gather {
            input: fuser.input_indexed(table).unwrap(),
            indices: fuser.input_indexed(&indices).unwrap(),
            output: fuser.output(&out).unwrap(),
            dim: 0,
        };
        fuser.fuse_operation(op);
        vec![indices, out]
    }

    /// Selects 4 rows of the table, and concatenates the table with itself.
    fn select_and_cat(fuser: &mut TraceFuser, table: &TensorIr) -> Vec<TensorIr> {
        let mut registered = select(fuser, table);
        let out = tensor(3, &[table.shape[0] * 2, COLS], DType::F32);
        let op = FuseOp::Cat {
            inputs: vec![fuser.input_indexed(table).unwrap(); 2],
            output: fuser.output(&out).unwrap(),
            dim: 0,
        };
        fuser.fuse_operation(op);
        registered.push(out);
        registered
    }

    #[test]
    fn a_select_reads_one_row_of_its_table() {
        let expected = COLS * F32 + 4 * I32 + 4 * COLS * F32;

        assert_eq!(traffic(4, select), expected);
        assert_eq!(traffic(100_000, select), expected);
    }

    #[test]
    fn a_select_of_no_row_reads_nothing_of_its_table() {
        assert_eq!(traffic(4, select_none), 0);
    }

    #[test]
    fn a_gather_reads_one_element_per_column_of_its_table() {
        let expected = COLS * F32 + 4 * COLS * I32 + 4 * COLS * F32;

        assert_eq!(traffic(4, gather), expected);
        assert_eq!(traffic(100_000, gather), expected);
    }

    #[test]
    fn a_table_a_concatenation_also_reads_is_read_whole() {
        let rows = 1_000;
        let expected = rows * COLS * F32 + 4 * I32 + 4 * COLS * F32 + 2 * rows * COLS * F32;

        assert_eq!(traffic(rows, select_and_cat), expected);
    }
}
