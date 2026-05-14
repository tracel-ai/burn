use alloc::format;
use burn_tensor::kind::{Autodiff, Basic};
use core::fmt::Display;

use crate as burn;
use crate::{
    module::{
        AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault,
        ModuleMapper, ModuleVisitor,
    },
    record::{PrecisionSettings, Record},
};
use burn_tensor::{Device, Tensor};

#[deprecated(
    since = "0.21.0",
    note = "ConstantRecord is misleading as it doesn't persist data. Use EmptyRecord instead."
)]
/// A record representing the absence of persistent module state.
pub type ConstantRecord = EmptyRecord;

/// A record representing the absence of persistent module state.
///
/// `EmptyRecord` is used for modules that do not store any data to be
/// serialized or restored (e.g., modules marked with `#[module(skip)]`
/// or modules without parameters).
///
/// This record contains no fields and serializes to `None`.
#[derive(Debug, Clone, Copy, new, Default, PartialEq, Eq)]
pub struct EmptyRecord;

impl serde::Serialize for EmptyRecord {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // nothing to serialize
        S::serialize_none(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for EmptyRecord {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_option(serde::de::IgnoredAny).ok();
        Ok(EmptyRecord::new())
    }
}

impl Record for EmptyRecord {
    type Item<S: PrecisionSettings> = EmptyRecord;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &Device) -> Self {
        item
    }
}
/// Constant macro.
#[macro_export]
macro_rules! empty {
    (module) => {
        type Record = burn::module::EmptyRecord;

        fn visit<V: burn::module::ModuleVisitor>(&self, _visitor: &mut V) {
            // Nothing to do
        }

        fn map<M: burn::module::ModuleMapper>(self, _mapper: &mut M) -> Self {
            self
        }

        fn load_record(self, _record: Self::Record) -> Self {
            self
        }

        fn into_record(self) -> Self::Record {
            burn::module::EmptyRecord::new()
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
    };

    (ad_module, $type:ty) => {
        fn valid(&self) -> Self {
            self.clone()
        }

        fn from_inner(module: Self) -> Self {
            module
        }
    };

    ($type:ty) => {
        impl burn::module::Module for $type {
            empty!(module);
        }

        impl burn::module::AutodiffModule for $type {
            empty!(ad_module, $type);
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
impl<const D: usize, K: Basic> Module for Tensor<D, K> {
    type Record = EmptyRecord;

    fn visit<V: ModuleVisitor>(&self, _visitor: &mut V) {}

    fn map<M: ModuleMapper>(self, _mapper: &mut M) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        EmptyRecord
    }

    fn load_record(self, _record: Self::Record) -> Self {
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
}

impl<const D: usize, K: Basic> ModuleDisplayDefault for Tensor<D, K> {
    fn content(&self, content: Content) -> Option<Content> {
        let string = format!("Tensor {{rank: {D}, shape: {:?}}}", self.shape().as_slice());
        content.add_single(&string).optional()
    }
}

impl<const D: usize, K: Basic> ModuleDisplay for Tensor<D, K> {}

impl<const D: usize, K: Autodiff> AutodiffModule for Tensor<D, K> {
    fn valid(&self) -> Self {
        self.clone().inner()
    }

    fn from_inner(tensor: Self) -> Self {
        Tensor::from_inner(tensor)
    }
}

// TODO: no longer necessary?
// impl<T> Module for PhantomData<T> {
//     type Record = EmptyRecord;

//     fn visit<V: ModuleVisitor>(&self, _visitor: &mut V) {
//         // Nothing to do
//     }

//     fn map<M: ModuleMapper>(self, _mapper: &mut M) -> Self {
//         self
//     }

//     fn load_record(self, _record: Self::Record) -> Self {
//         self
//     }

//     fn into_record(self) -> Self::Record {
//         EmptyRecord::new()
//     }

//     fn to_device(self, _: &Device) -> Self {
//         self
//     }

//     fn fork(self, _: &Device) -> Self {
//         self
//     }

//     fn collect_devices(&self, devices: Devices) -> Devices {
//         devices
//     }
// }

// impl<T> ModuleDisplayDefault for PhantomData<T> {
//     fn content(&self, content: Content) -> Option<Content> {
//         content.add_single(&"PhantomData".to_string()).optional()
//     }
// }

// impl<T> ModuleDisplay for PhantomData<T> {}

// impl<T> AutodiffModule for PhantomData<T> {
//     fn valid(&self) -> Self {
//         PhantomData
//     }

//     fn from_inner(_module: Self) -> Self {
//         Self
//     }
// }

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
    type Record = EmptyRecord;

    fn visit<V: ModuleVisitor>(&self, _visitor: &mut V) {
        // Nothing to do
    }

    fn map<M: ModuleMapper>(self, _mapper: &mut M) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        EmptyRecord::new()
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
impl<T> AutodiffModule for Ignored<T>
where
    T: Sync + Send + core::fmt::Debug + Clone,
{
    fn valid(&self) -> Self {
        self.clone()
    }

    fn from_inner(module: Self) -> Self {
        module
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

    #[cfg(feature = "autodiff")]
    #[test]
    fn tensor_load_record_setting() {
        use crate::{
            TestDevice,
            record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        };
        use burn_tensor::{Device, Tensor};

        let device = &Device::new(TestDevice::default()).autodiff();
        let tensor = Tensor::<2>::ones([3, 3], device);

        let byte_recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = Recorder::record(&byte_recorder, tensor.clone().into_record(), ()).unwrap();

        let no_grad_is_require_grad = tensor
            .clone()
            .no_grad()
            .load_record(Recorder::load(&byte_recorder, bytes.clone(), device).unwrap())
            .is_require_grad();

        let with_default_is_require_grad = tensor
            .load_record(Recorder::load(&byte_recorder, bytes.clone(), device).unwrap())
            .is_require_grad();

        assert!(!no_grad_is_require_grad);
        assert!(!with_default_is_require_grad);
    }

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
