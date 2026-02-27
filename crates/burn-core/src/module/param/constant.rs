use alloc::{format, string::ToString};
use core::{fmt::Display, marker::PhantomData};
use serde::{Serialize, de::DeserializeOwned};

use crate as burn;
use crate::{
    module::{
        AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault,
        ModuleMapper, ModuleVisitor,
    },
    record::{PrecisionSettings, Record},
};
use burn_tensor::{
    BasicAutodiffOps, BasicOps, Tensor,
    backend::{AutodiffBackend, Backend},
    ops::Device,
};

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

impl<B: Backend> Record<B> for EmptyRecord {
    type Item<S: PrecisionSettings> = EmptyRecord;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        item
    }
}
/// Constant macro.
#[macro_export]
macro_rules! empty {
    (module) => {
        type Record = burn::module::EmptyRecord;

        fn visit<V: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut V) {
            // Nothing to do
        }

        fn map<M: burn::module::ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
            self
        }

        fn load_record(self, _record: Self::Record) -> Self {
            self
        }

        fn into_record(self) -> Self::Record {
            burn::module::EmptyRecord::new()
        }

        fn to_device(self, _: &B::Device) -> Self {
            self
        }

        fn fork(self, _: &B::Device) -> Self {
            self
        }

        fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
            devices
        }
    };

    (ad_module, $type:ty) => {
        type InnerModule = $type;

        fn valid(&self) -> Self::InnerModule {
            self.clone()
        }

        fn from_inner(module: Self::InnerModule) -> Self {
            module
        }
    };

    ($type:ty) => {
        impl<B: burn::tensor::backend::Backend> burn::module::Module<B> for $type {
            empty!(module);
        }

        impl<B: burn::tensor::backend::AutodiffBackend> burn::module::AutodiffModule<B> for $type {
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

// TODO: breaking change for these constant types (currently empty record, non-persistent)

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
impl<const D: usize, B: Backend, K: BasicOps<B>> Module<B> for Tensor<B, D, K> {
    type Record = EmptyRecord;

    fn visit<V: ModuleVisitor<B>>(&self, _visitor: &mut V) {}

    fn map<M: ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        EmptyRecord
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn to_device(self, device: &B::Device) -> Self {
        self.to_device(device)
    }

    fn fork(self, device: &B::Device) -> Self {
        self.to_device(device)
    }

    fn collect_devices(&self, mut devices: Devices<B>) -> Devices<B> {
        let device = self.device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<const D: usize, B: Backend, K: BasicOps<B>> ModuleDisplayDefault for Tensor<B, D, K> {
    fn content(&self, content: Content) -> Option<Content> {
        let string = format!("Tensor {{rank: {D}, shape: {:?}}}", self.shape().as_slice());
        content.add_single(&string).optional()
    }
}

impl<const D: usize, B: Backend, K: BasicOps<B>> ModuleDisplay for Tensor<B, D, K> {}

impl<const D: usize, B: AutodiffBackend, K: BasicAutodiffOps<B>> AutodiffModule<B>
    for Tensor<B, D, K>
{
    type InnerModule = Tensor<B::InnerBackend, D, K::InnerKind>;

    fn valid(&self) -> Self::InnerModule {
        self.clone().inner()
    }

    fn from_inner(tensor: Self::InnerModule) -> Self {
        Tensor::from_inner(tensor)
    }
}

impl<B: Backend> Module<B> for PhantomData<B> {
    type Record = EmptyRecord;

    fn visit<V: ModuleVisitor<B>>(&self, _visitor: &mut V) {
        // Nothing to do
    }

    fn map<M: ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        EmptyRecord::new()
    }

    fn to_device(self, _: &Device<B>) -> Self {
        self
    }

    fn fork(self, _: &Device<B>) -> Self {
        self
    }

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }
}

impl<B: Backend> ModuleDisplayDefault for PhantomData<B> {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_single(&"PhantomData".to_string()).optional()
    }
}

impl<B: Backend> ModuleDisplay for PhantomData<B> {}

impl<B: AutodiffBackend> AutodiffModule<B> for PhantomData<B> {
    type InnerModule = PhantomData<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        PhantomData
    }

    fn from_inner(_module: Self::InnerModule) -> Self {
        PhantomData
    }
}

/// Container to satisfy the Module trait for types that are not modules.
#[derive(Clone, Debug)]
#[deprecated(
    since = "0.21.0",
    note = "Ignored<T> is deprecated. Use #[module(skip)] for non-persistent fields (same behavior) or #[module(constant)] for persistent fields."
)]
pub struct Ignored<T>(pub T);

#[allow(deprecated)]
impl<B, T> Module<B> for Ignored<T>
where
    B: Backend,
    T: Sync + Send + core::fmt::Debug + Clone,
{
    type Record = EmptyRecord;

    fn visit<V: ModuleVisitor<B>>(&self, _visitor: &mut V) {
        // Nothing to do
    }

    fn map<M: ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        EmptyRecord::new()
    }

    fn to_device(self, _: &Device<B>) -> Self {
        self
    }

    fn fork(self, _: &Device<B>) -> Self {
        self
    }

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
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
impl<B: AutodiffBackend, T> AutodiffModule<B> for Ignored<T>
where
    B: AutodiffBackend,
    T: Sync + Send + core::fmt::Debug + Clone,
{
    type InnerModule = Ignored<T>;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
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

/// A record that persists a simple value.
///
/// This is used for fields marked with `#[module(constant)]`.
#[derive(Debug, Clone, Copy, new)]
pub struct ValueRecord<T> {
    value: T,
}

impl<T: PartialEq> PartialEq for ValueRecord<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T> ValueRecord<T> {
    /// Gets the value while consuming the record.
    pub fn consume(self) -> T {
        self.value
    }
}

impl<B: Backend, T> Record<B> for ValueRecord<T>
where
    T: Send + Serialize + DeserializeOwned + Clone, // Record and item
{
    // The Item is the type T itself, as it already satisfies the bounds
    type Item<S: PrecisionSettings> = T;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.value
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        Self { value: item }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use core::marker::PhantomData;

    use burn_tensor::backend::Backend;
    use burn_tensor::{Device, Tensor};

    use crate::TestBackend;
    use crate::{
        TestAutodiffBackend,
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    };
    use burn::module::Module;

    use crate as burn;

    #[test]
    fn tensor_load_record_setting() {
        let device: &Device<TestAutodiffBackend> = &Default::default();
        let tensor = Tensor::<TestAutodiffBackend, 2>::ones([3, 3], device);

        let byte_recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = Recorder::<TestAutodiffBackend>::record(
            &byte_recorder,
            tensor.clone().into_record(),
            (),
        )
        .unwrap();

        let no_grad_is_require_grad = tensor
            .clone()
            .no_grad()
            .load_record(
                Recorder::<TestAutodiffBackend>::load(&byte_recorder, bytes.clone(), device)
                    .unwrap(),
            )
            .is_require_grad();

        let with_default_is_require_grad = tensor
            .load_record(
                Recorder::<TestAutodiffBackend>::load(&byte_recorder, bytes.clone(), device)
                    .unwrap(),
            )
            .is_require_grad();

        assert!(!no_grad_is_require_grad);
        assert!(!with_default_is_require_grad);
    }

    #[test]
    fn empty_module_with_phantom() {
        #[derive(Module, Debug, new)]
        struct EmptyModule<B: Backend> {
            _phantom: PhantomData<B>,
        }

        let _module = EmptyModule::<TestBackend>::new();

        assert_eq!(core::mem::size_of::<EmptyModule<TestBackend>>(), 0);
    }
}
