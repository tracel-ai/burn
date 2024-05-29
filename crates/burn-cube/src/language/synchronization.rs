use crate::{dialect::Synchronization, CubeContext};

pub fn sync_units() {}

pub fn sync_units_expand(context: &mut CubeContext) {
    context.register(Synchronization::WorkgroupBarrier)
}