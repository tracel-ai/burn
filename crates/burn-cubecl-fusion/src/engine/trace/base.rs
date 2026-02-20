use crate::engine::{
    codegen::ir::{FuseArg, FuseType},
    trace::block::FuseBlock,
};
use burn_ir::{TensorId, TensorIr};
use burn_std::{Shape, Strides};
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet},
    marker::PhantomData,
};

#[cfg(feature = "autotune-checks")]
use crate::CubeFusionHandle;
#[cfg(feature = "autotune-checks")]
use burn_backend::TensorData;
#[cfg(feature = "autotune-checks")]
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize, Debug)]
/// A trace contains all [blocks](FuseBlock) and the [resources](FuseResources) used by the
/// kernel.
pub struct FuseTrace {
    pub blocks: Vec<FuseBlock>,
    pub resources: FuseResources,
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

pub enum TuneOutput<R: Runtime> {
    UnChecked(PhantomData<R>),
    #[cfg(feature = "autotune-checks")]
    Checked {
        handles: HashMap<TensorId, (Vec<usize>, CubeFusionHandle<R>)>,
    },
}

impl<R: Runtime> TuneOutput<R> {
    #[allow(unused_variables)]
    pub fn merge(self, other: Self) -> Self {
        let mut result = self;

        match &mut result {
            TuneOutput::UnChecked(..) => {}
            #[cfg(feature = "autotune-checks")]
            TuneOutput::Checked { handles } => match other {
                TuneOutput::UnChecked(..) => {}
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

impl<R: Runtime> cubecl::tune::AutotuneOutput for TuneOutput<R> {
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
                    use cubecl::std::tensor::into_contiguous_ref;

                    let current_handle = if !is_contiguous(&shape, &handle.strides) {
                        into_contiguous_ref::<R>(
                            &handle.client,
                            &handle.as_handle_ref(&shape),
                            handle.dtype.into(),
                        )
                        .unwrap()
                        .handle
                    } else {
                        handle.handle.clone()
                    };
                    let other_handle = if !is_contiguous(&shape, &other.strides) {
                        into_contiguous_ref::<R>(
                            &other.client,
                            &other.as_handle_ref(&shape),
                            other.dtype.into(),
                        )
                        .unwrap()
                        .handle
                    } else {
                        other.handle.clone()
                    };

                    let data_ref = handle.client.read_one(current_handle);
                    let data_other = other.client.read_one(other_handle);
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
/// Declare all resources used by the kernel, and potentially multiple [blocks](FuseBlock).
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

impl RegisteredTensors {
    /// Iterate over all the registered tensors.
    pub fn iter(&self) -> impl Iterator<Item = &RegisterTensor> {
        self.tensors.iter()
    }

    /// Consumes and iterate over all the registered tensors.
    pub fn into_iter(self) -> impl Iterator<Item = RegisterTensor> {
        self.tensors.into_iter()
    }

    /// Returns the number of tensors registered.
    pub fn len(&self) -> usize {
        self.tensors.len()
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

    /// Insert a normal tensor with the given [precision](FusePrecision) in the current block.
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
