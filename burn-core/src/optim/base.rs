use crate::module::ADModule;
use crate::record::Record;
use crate::tensor::backend::ADBackend;

use super::GradientsParams;

pub trait Optimizer<M, B>: Send + Sync
where
    M: ADModule<B>,
    B: ADBackend,
{
    type Record: Record;

    fn step(&mut self, module: M, grads: GradientsParams) -> M;
    fn to_record(&self) -> Self::Record;
    fn load_record(self, record: Self::Record) -> Self;
}
