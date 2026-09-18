use alloc::format;
use burn_tensor::kind::{Autodiff, Basic};
use core::fmt::Display;

use crate as burn;
use crate::module::{
    Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper, ModuleVisitor,
};
use burn_tensor::{Device, Tensor};

/// Constant macro.
#[macro_export]
macro_rules! empty {
    (module) => {
        fn visit<V: burn::module::ModuleVisitor>(&self, _visitor: &mut V) {
            // Nothing to do
        }

        fn map<M: burn::module::ModuleMapper>(self, _mapper: &mut M) -> Self {
            self
        }

        fn to_device(self, _: &burn::tensor::Device) -> Self {
            self
        }

        fn fork(self, _: &burn::tensor::Device) -> Self {
            self
        }

        fn collect_devices(&self, devices: burn::module::Devices) -> burn::module::Devices {
            devices
        }

        fn valid(&self) -> Self {
            self.clone()
        }

        fn train(self) -> Self {
            self
        }
    };

    ($type:ty) => {
        impl burn::module::Module for $type {
            empty!(module);
        }

        impl burn::module::ModuleDisplayDefault for $type {
            fn content(&self, content: burn::module::Content) -> Option<burn::module::Content> {
                let string = format!("{}", self);
                content.add_formatted(&string).optional()
            }
        }

        impl burn::module::ModuleDisplay for $type {}
    };
}

// TODO: breaking change for these constant types (currently empty record, non-persistent)?

// General Types
empty!(alloc::string::String);
empty!(bool);

// Float Types
empty!(f64);
empty!(f32);
empty!(half::bf16);
empty!(half::f16);

// Unsigned Integer Types
empty!(usize);
empty!(u64);
empty!(u32);
empty!(u16);
empty!(u8);

// Signed Integer Types
empty!(isize);
empty!(i64);
empty!(i32);
empty!(i16);
empty!(i8);

impl burn::module::ModuleDisplay for str {}
impl burn::module::ModuleDisplayDefault for str {
    fn content(&self, content: burn::module::Content) -> Option<burn::module::Content> {
        content.add_formatted(&self).optional()
    }
}

// TODO: tensor record should persist
impl<const D: usize, K: Autodiff> Module for Tensor<D, K> {
    fn visit<V: ModuleVisitor>(&self, _visitor: &mut V) {}

    fn map<M: ModuleMapper>(self, _mapper: &mut M) -> Self {
        self
    }

    fn to_device(self, device: &Device) -> Self {
        self.to_device(device)
    }

    fn fork(self, device: &Device) -> Self {
        self.to_device(device)
    }

    fn collect_devices(&self, mut devices: Devices) -> Devices {
        let device = self.device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }

    fn valid(&self) -> Self {
        self.clone().without_autodiff()
    }

    fn train(self) -> Self {
        Tensor::from_inner(self)
    }
}

impl<const D: usize, K: Basic> ModuleDisplayDefault for Tensor<D, K> {
    fn content(&self, content: Content) -> Option<Content> {
        let string = format!("Tensor {{rank: {D}, shape: {:?}}}", self.shape().as_slice());
        content.add_single(&string).optional()
    }
}

impl<const D: usize, K: Basic> ModuleDisplay for Tensor<D, K> {}

/// Container to satisfy the Module trait for types that are not modules.
#[derive(Clone, Debug)]
#[deprecated(
    since = "0.21.0",
    note = "Ignored<T> is deprecated. Use #[module(skip)] for non-persistent fields (same behavior)."
)]
pub struct Ignored<T>(pub T);

#[allow(deprecated)]
impl<T> Module for Ignored<T>
where
    T: Sync + Send + core::fmt::Debug + Clone,
{
    fn visit<V: ModuleVisitor>(&self, _visitor: &mut V) {
        // Nothing to do
    }

    fn map<M: ModuleMapper>(self, _mapper: &mut M) -> Self {
        self
    }

    fn to_device(self, _: &Device) -> Self {
        self
    }

    fn fork(self, _: &Device) -> Self {
        self
    }

    fn collect_devices(&self, devices: Devices) -> Devices {
        devices
    }

    fn valid(&self) -> Self {
        self.clone()
    }

    fn train(self) -> Self {
        self
    }
}

#[allow(deprecated)]
impl<T> ModuleDisplayDefault for Ignored<T>
where
    T: Sync + Send + core::fmt::Debug + Clone,
{
    fn content(&self, content: Content) -> Option<Content> {
        // For now, just print the debug representation of the ignored value
        content.add_single(&format!("{:?}", self.0)).optional()
    }
}

#[allow(deprecated)]
impl<T> ModuleDisplay for Ignored<T> where T: Sync + Send + core::fmt::Debug + Clone {}

#[allow(deprecated)]
impl<T> Display for Ignored<T>
where
    T: Sync + Send + core::fmt::Debug + Clone,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

#[allow(deprecated)]
// Implement deref for Ignored
impl<T> core::ops::Deref for Ignored<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use core::marker::PhantomData;

    use burn::module::Module;

    use crate as burn;

    #[test]
    fn empty_module_with_phantom() {
        #[derive(Module, Debug, new)]
        struct EmptyModule<T: core::fmt::Debug + Clone + Send> {
            #[module(skip)]
            _phantom: PhantomData<T>,
        }

        let _module = EmptyModule::<bool>::new();

        assert_eq!(core::mem::size_of::<EmptyModule<bool>>(), 0);
    }
}
