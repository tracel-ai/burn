use core::marker::PhantomData;

use burn_tensor::{backend::Backend, benchmark::Benchmark};
use hashbrown::HashMap;

use crate::server::ComputeServer;

use super::{InputHashable, Operation};

pub(crate) struct KernelType<B, BM, S>
where
    B: Backend,
    BM: Benchmark<B>,
{
    _backend: PhantomData<B>,
    _benchmark: PhantomData<BM>,
    _server: PhantomData<S>,
}

impl<B, BM, S> Clone for KernelType<B, BM, S>
where
    B: Backend,
    BM: Benchmark<B>,
{
    fn clone(&self) -> KernelType<B, BM, S> {
        KernelType {
            _backend: PhantomData,
            _benchmark: PhantomData,
            _server: PhantomData,
        }
    }
}
impl<B, BM, S> Copy for KernelType<B, BM, S>
where
    B: Backend,
    BM: Benchmark<B>,
{
}

impl<B, BM, S: ComputeServer> KernelType<B, BM, S>
where
    B: Backend,
    BM: Benchmark<B>,
{
    pub(crate) fn to_kernel(&self) -> S::Kernel {
        todo!()
    }
    pub(crate) fn to_benchmark(&self) -> BM {
        todo!()
    }
}

pub(crate) struct KernelPool<B, BM, O: Operation<S>, S: ComputeServer>
where
    B: Backend,
    BM: Benchmark<B>,
{
    cache: HashMap<String, usize>,
    pub kernel_types: Vec<KernelType<B, BM, S>>, // is actually static?
    _operation: PhantomData<O>,
}

impl<B, BM, O: Operation<S>, S: ComputeServer> KernelPool<B, BM, O, S>
where
    B: Backend,
    BM: Benchmark<B>,
{
    pub(crate) fn try_cache(&self, input: &O::Input) -> Option<KernelType<B, BM, S>> {
        let index = self.cache.get(&input.custom_hash());
        index.map(|i| self.kernel_types[*i])
    }

    pub(crate) fn get(&self, index: usize) -> KernelType<B, BM, S> {
        *self.kernel_types.get(index).unwrap()
    }

    pub(crate) fn add_to_cache(&mut self, input: &O::Input, index: usize) -> () {
        self.cache.insert(input.custom_hash(), index);
    }
}
