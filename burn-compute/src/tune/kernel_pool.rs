use core::marker::PhantomData;

use hashbrown::HashMap;

use crate::server::ComputeServer;

use super::{InputHashable, Operation};

pub(crate) struct KernelType<S> {
    _server: PhantomData<S>,
}

impl<S> Clone for KernelType<S> {
    fn clone(&self) -> KernelType<S> {
        KernelType {
            _server: PhantomData,
        }
    }
}

impl<S> Copy for KernelType<S> {}

impl<S: ComputeServer> KernelType<S> {
    pub(crate) fn to_kernel(&self) -> S::Kernel {
        todo!()
    }
}

pub(crate) struct KernelPool<O: Operation<S>, S: ComputeServer> {
    cache: HashMap<String, usize>,
    pub kernel_types: Vec<KernelType<S>>, // is actually static?
    _operation: PhantomData<O>,
}

impl<O: Operation<S>, S: ComputeServer> KernelPool<O, S> {
    pub(crate) fn try_cache(&self, input: &O::Input) -> Option<KernelType<S>> {
        let index = self.cache.get(&input.custom_hash());
        index.map(|i| self.kernel_types[*i])
    }

    pub(crate) fn get(&self, index: usize) -> KernelType<S> {
        *self.kernel_types.get(index).unwrap()
    }

    pub(crate) fn add_to_cache(&mut self, input: &O::Input, index: usize) -> () {
        self.cache.insert(input.custom_hash(), index);
    }
}
