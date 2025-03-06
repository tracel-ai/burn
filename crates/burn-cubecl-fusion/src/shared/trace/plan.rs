use cubecl::prelude::Sequence;
use std::collections::BTreeMap;

use crate::{
    shared::ir::{Arg, ElemwiseOp, ElemwisePrecision},
    CubeFusionHandle,
};
use burn_ir::{TensorId, TensorIr};
use cubecl::Runtime;

/// The plan is responsible to keep runtime information related to the launch of a fused kernel
/// at one place.
#[derive(Debug)]
pub(crate) struct LaunchPlan<'a, R: Runtime> {
    pub potential_inplaces: Vec<PotentialInplace<'a>>,
    pub potential_reference_input: Option<usize>,
    pub potential_reference_reshape: Option<Sequence<Arg>>,
    pub global_inputs: Vec<TensorIr>,
    pub global_outputs: Vec<TensorIr>,
    pub handle_inputs: Vec<HandleInput<R>>,
    pub handle_outputs: Vec<HandleOutput<R>>,
    pub reference: ReferenceSelection,
    pub reads: BTreeMap<TensorId, Vec<ElemwiseOp>>,
    pub writes: BTreeMap<TensorId, ElemwiseOp>,
    pub vectorization: BTreeMap<TensorId, Vect>,
    pub width: u8,
    pub rank: usize,
}

#[derive(Debug)]
pub enum ReferenceSelection {
    Searching,
    NotFound,
    Found(Reference),
    Virtual(Sequence<Arg>),
}

impl ReferenceSelection {
    pub fn is_found(&self) -> bool {
        matches!(self, Self::Found(..))
    }

    pub fn compatible_strides_for_inplace(&self, strides: &[usize]) -> bool {
        match self {
            ReferenceSelection::Found(reference) => reference.strides == strides,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Vect {
    Broadcasted,
    Aligned(u8),
}

impl Vect {
    pub fn line_size(&self) -> u8 {
        match self {
            Vect::Broadcasted => 1,
            Vect::Aligned(val) => *val,
        }
    }

    pub fn is_broadcast(&self) -> bool {
        matches!(self, Vect::Broadcasted)
    }

    pub fn limit_to_one(&self) -> Self {
        match self {
            Vect::Broadcasted => Vect::Broadcasted,
            Vect::Aligned(_) => Vect::Aligned(1),
        }
    }
}

impl<R: Runtime> LaunchPlan<'_, R> {
    pub fn output_offset(&mut self, output_offset: u32) {
        if let ReferenceSelection::Found(re) = &mut self.reference {
            if let Arg::Output(pos, ..) = &mut re.layout {
                *pos += output_offset
            }
        }

        for op in self.writes.iter_mut() {
            op.1.output_offset(output_offset);
        }
    }

    pub fn new(
        reads: &BTreeMap<TensorId, Vec<ElemwiseOp>>,
        writes: &BTreeMap<TensorId, ElemwiseOp>,
        rank: usize,
    ) -> Self {
        LaunchPlan {
            potential_inplaces: Vec::new(),
            potential_reference_input: None,
            potential_reference_reshape: None,
            global_inputs: Vec::new(),
            global_outputs: Vec::new(),
            handle_inputs: Vec::new(),
            handle_outputs: Vec::new(),
            reference: ReferenceSelection::Searching,
            vectorization: BTreeMap::default(),
            reads: reads.clone(),
            writes: writes.clone(),
            width: 1,
            rank,
        }
    }
}

#[derive(Debug)]
pub enum HandleOutput<R: Runtime> {
    Alias {
        input_pos: usize,
        precision: ElemwisePrecision,
    },
    Owned {
        global_id: TensorId,
        precision: ElemwisePrecision,
        handle: CubeFusionHandle<R>,
        global_shape: Vec<usize>,
        vectorization: u8,
    },
}

#[derive(Debug)]
pub struct HandleInput<R: Runtime> {
    pub relative_id: TensorId,
    pub global_id: TensorId,
    pub precision: ElemwisePrecision,
    pub handle: CubeFusionHandle<R>,
    pub global_shape: Vec<usize>,
    pub vectorization: u8,
    pub broadcated: bool,
}

#[derive(Debug)]
pub struct Reference {
    pub layout: Arg,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

#[derive(Debug)]
pub struct PotentialInplace<'a> {
    pub input_pos: usize,
    pub tensor_relative: &'a TensorIr,
    pub strides: Vec<usize>,
}
