use cubecl::prelude::*;

use crate::JitElement;

#[cube]
pub trait ReduceDimSubcube<EIn: JitElement, EOut: JitElement>: Send + Sync + 'static {
    type Accumulator: CubeType;
    type Value: CubeType;

    fn init_shared(#[comptime] size: u32) -> Self::Accumulator;
    fn init_value() -> Self::Value;
    fn read_value(input: &Tensor<EIn>, pos: u32, i: u32) -> Self::Value;
    fn read_from_shared(acc: &Self::Accumulator, pos: u32) -> Self::Value;
    fn update_value(current: &mut Self::Value, new: Self::Value);
    fn reduce_subcube(acc: &mut Self::Accumulator, pos: u32, value: Self::Value);
    fn store(acc: &Self::Accumulator, out: &mut Tensor<EOut>, pos: u32, dim_len: u32);
}
