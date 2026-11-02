//! Walking a storage-tiled input in its tiles.
//!
//! A storage-tiled tensor is a strided tensor with more dims: each logical dim split into the
//! fragments its tiles store it as. An elementwise block reads and writes each element once and
//! never asks for its coordinates, so it can walk the tiles' physical dims instead of the logical
//! ones. The tiled input is read as it lies, every other input is split by strides into the same
//! dims, which moves no data, and the outputs are written in the tiles and handed back tiled.

use crate::{
    CubeFusionHandle, Tiles,
    engine::{
        codegen::ir::FuseOp,
        launch::layout::{dim_order, strides_for},
        settings::RefLayoutSetting,
        trace::{FuseResources, RegisterTensor, block::FuseBlock},
    },
    strides_dyn_rank,
};
use burn_fusion::stream::Context;
use burn_std::{Metadata, Shape, Strides};

/// The storage tiles a trace's blocks walk instead of their logical dims.
#[derive(Debug)]
pub struct StorageTiling {
    /// How the tiled input is stored.
    stored: Metadata,
    /// Its logical shape, every block's.
    logical: Shape,
}

impl StorageTiling {
    /// The tiles the trace walks, or `None` where it walks its logical dims: no input is a float
    /// stored in tiles at the blocks' shape, or a block does more than elementwise work.
    pub fn new<'r>(
        resources: &'r FuseResources,
        blocks: &[FuseBlock],
        context: &Context<CubeFusionHandle>,
    ) -> Option<Self> {
        let shape_ref = &blocks.first()?.shape_ref;
        let at_shape_ref = |tensor: &'r RegisterTensor| match tensor {
            RegisterTensor::Normal(tensor, _) => Some(tensor).filter(|t| &t.shape == shape_ref),
            RegisterTensor::QuantValues(_) | RegisterTensor::QuantParams(_) => None,
        };

        let elementwise = blocks
            .iter()
            .all(|block| &block.shape_ref == shape_ref && walks_any_layout(block))
            && resources.views.is_empty()
            && resources.indexed.is_empty()
            && resources.inputs_unhandled.is_empty()
            && resources.outputs_unhandled.is_empty()
            && resources
                .outputs
                .iter()
                .all(|output| at_shape_ref(output).is_some())
            && resources
                .inputs
                .iter()
                .all(|input| matches!(input, RegisterTensor::Normal(..)));
        if !elementwise {
            return None;
        }

        resources
            .inputs
            .iter()
            .filter_map(at_shape_ref)
            .find_map(|input| {
                let global = context.tensors.get(&input.id)?;
                let handle = context.handles.get_handle_ref(&global.id)?;
                (handle.tiles.is_some() && handle.qparams.is_none()).then(|| Self {
                    stored: handle.metadata(global.shape.clone()),
                    logical: global.shape.clone(),
                })
            })
    }

    /// The physical shape every tensor of the trace is walked in.
    pub fn shape(&self) -> &Shape {
        &self.stored.shape
    }

    /// Whether `handle` is stored in these tiles, so it is read as it lies.
    pub fn stores(&self, handle: &CubeFusionHandle) -> bool {
        handle.tiles.as_ref().is_some_and(|tiles| {
            tiles.tiling == self.stored.tiling && tiles.shape == self.stored.shape
        })
    }

    /// The strides an input of logical `shape` is walked with over the physical dims: its own
    /// where it is stored in these tiles, otherwise its rows broadcast to the trace's shape and
    /// split by strides.
    ///
    /// # Panics
    ///
    /// On an input stored in other tiles: it is laid back in rows first.
    pub fn strides(&self, handle: &CubeFusionHandle, shape: &Shape) -> Strides {
        if self.stores(handle) {
            return handle.strides.clone();
        }
        assert!(
            handle.tiles.is_none(),
            "an input stored in other tiles is laid back in rows first"
        );
        let leading = self.logical.rank() - shape.rank();
        let broadcast = self
            .logical
            .iter()
            .enumerate()
            .map(|(dim, &extent)| match dim.checked_sub(leading) {
                Some(dim) if shape[dim] == extent => handle.strides[dim],
                _ => 0,
            })
            .collect::<Vec<_>>();
        Metadata::new(self.logical.clone(), Strides::new(&broadcast))
            .tiled_like(&self.stored)
            .expect("an input broadcast to the trace's shape splits into its tiles")
            .strides()
            .clone()
    }

    /// The strides an output is allocated with: the tiled input's order over the physical dims,
    /// so the output is stored as it is.
    pub fn output_strides(&self) -> Strides {
        match dim_order(&self.stored.shape, self.stored.strides()) {
            Some(order) => strides_for(&self.stored.shape, &order),
            None => strides_dyn_rank(&self.stored.shape),
        }
    }

    /// The tiles an output written over the physical dims is stored in.
    pub fn tiles(&self) -> Tiles {
        Tiles {
            shape: self.stored.shape.clone(),
            tiling: self.stored.tiling,
        }
    }
}

/// Whether a block reads and writes every operand through its reference alone, so it can walk
/// any layout: it may choose its own, and no operation indexes an operand by its coordinates.
fn walks_any_layout(block: &FuseBlock) -> bool {
    matches!(block.settings.ref_layout, RefLayoutSetting::Any)
        && block.settings.choose_output_layout
        && block.ops.iter().all(|op| {
            !matches!(
                op,
                FuseOp::Gather { .. }
                    | FuseOp::Select { .. }
                    | FuseOp::Cat { .. }
                    | FuseOp::Dequantize { .. }
            )
        })
}
