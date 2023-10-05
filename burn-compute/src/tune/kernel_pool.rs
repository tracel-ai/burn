use core::marker::PhantomData;

use hashbrown::HashMap;

use crate::{channel::ComputeChannel, server::ComputeServer};

use super::{Benchmark, InputHashable, Operation};

pub struct KernelType<O, S, C, BM> {
    _server: PhantomData<S>,
    _operation: PhantomData<O>,
    _channel: PhantomData<C>,
    _benchmark: PhantomData<BM>,
}

impl<O, S, C, BM> Clone for KernelType<O, S, C, BM> {
    fn clone(&self) -> KernelType<O, S, C, BM> {
        KernelType {
            _server: PhantomData,
            _operation: PhantomData,
            _channel: PhantomData,
            _benchmark: PhantomData,
        }
    }
}
impl<O, S, C, BM> Copy for KernelType<O, S, C, BM> {}

impl<O: Operation<S>, S: ComputeServer, C: ComputeChannel<S>, BM: Benchmark<O, S, C>>
    KernelType<O, S, C, BM>
{
    pub(crate) fn to_kernel(&self) -> S::Kernel {
        todo!()
    }
    pub(crate) fn to_benchmark(&self) -> BM {
        todo!()
    }
}

pub struct KernelPool<O: Operation<S>, S: ComputeServer, C, BM> {
    cache: HashMap<String, usize>,
    pub kernel_types: Vec<KernelType<O, S, C, BM>>, // is actually static?
    _operation: PhantomData<O>,
}

impl<O: Operation<S>, S: ComputeServer, C, BM> KernelPool<O, S, C, BM> {
    pub(crate) fn try_cache(&self, input: &O::Input) -> Option<KernelType<O, S, C, BM>> {
        let index = self.cache.get(&input.custom_hash());
        index.map(|i| self.kernel_types[*i])
    }

    pub(crate) fn get(&self, index: usize) -> KernelType<O, S, C, BM> {
        *self.kernel_types.get(index).unwrap()
    }

    pub(crate) fn add_to_cache(&mut self, input: &O::Input, index: usize) -> () {
        self.cache.insert(input.custom_hash(), index);
    }
}
