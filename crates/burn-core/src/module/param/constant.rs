use alloc::{format, string::ToString};
use core::{fmt::Display, marker::PhantomData};

use crate::{
    self as burn,
    module::{
        AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault,
        ModuleMapper, ModuleVisitor,
    },
    record::Record,
};
use burn::record::PrecisionSettings;
use burn_tensor::{
    backend::{AutodiffBackend, Backend},
    BasicAutodiffOps, BasicOps, Tensor,
};

/// Record used for constant type implementing the [module](crate::module::Module) trait.
#[derive(Debug, Clone, Copy, new, Default)]
pub struct ConstantRecord;

impl serde::Serialize for ConstantRecord {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // nothing to serialize
        S::serialize_none(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for ConstantRecord {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_option(serde::de::IgnoredAny).ok();
        Ok(ConstantRecord::new())
    }
}

impl<B: Backend> Record<B> for ConstantRecord {
    type Item<S: PrecisionSettings> = ConstantRecord;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        item
    }
}
/// Constant macro.
#[macro_export]
macro_rules! constant {
    (module) => {
        type Record = burn::module::ConstantRecord;

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
            burn::module::ConstantRecord::new()
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
    };

    ($type:ty) => {
        impl<B: burn::tensor::backend::Backend> burn::module::Module<B> for $type {
            constant!(module);
        }

        impl<B: burn::tensor::backend::AutodiffBackend> burn::module::AutodiffModule<B> for $type {
            constant!(ad_module, $type);
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

// General Types
constant!(alloc::string::String);
constant!(bool);

// Float Types
constant!(f64);
constant!(f32);
constant!(half::bf16);
constant!(half::f16);

// Unsigned Integer Types
constant!(usize);
constant!(u64);
constant!(u32);
constant!(u16);
constant!(u8);

// Signed Integer Types
constant!(i64);
constant!(i32);
constant!(i16);
constant!(i8);

impl burn::module::ModuleDisplay for str {}
impl burn::module::ModuleDisplayDefault for str {
    fn content(&self, content: burn::module::Content) -> Option<burn::module::Content> {
        content.add_formatted(&self).optional()
    }
}

impl<const D: usize, B: Backend, K: BasicOps<B>> Module<B> for Tensor<B, D, K> {
    type Record = ConstantRecord;

    fn visit<V: ModuleVisitor<B>>(&self, _visitor: &mut V) {}

    fn map<M: ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        ConstantRecord
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
        let string = format!("Tensor {{rank: {D}, shape: {:?}}}", self.shape().dims);
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
}

impl<B: Backend> Module<B> for PhantomData<B> {
    type Record = ConstantRecord;

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
        ConstantRecord::new()
    }

    fn to_device(self, _: &<B as Backend>::Device) -> Self {
        self
    }

    fn fork(self, _: &<B as Backend>::Device) -> Self {
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
}

/// Container to satisfy the Module trait for types that are not modules.
#[derive(Clone, Debug)]
pub struct Ignored<T>(pub T);

impl<B, T> Module<B> for Ignored<T>
where
    B: Backend,
    T: Sync + Send + core::fmt::Debug + Clone,
{
    type Record = ConstantRecord;

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
        ConstantRecord::new()
    }

    fn to_device(self, _: &<B as Backend>::Device) -> Self {
        self
    }

    fn fork(self, _: &<B as Backend>::Device) -> Self {
        self
    }

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }
}

impl<T> ModuleDisplayDefault for Ignored<T>
where
    T: Sync + Send + core::fmt::Debug + Clone,
{
    fn content(&self, content: Content) -> Option<Content> {
        // For now, just print the debug representation of the ignored value
        content.add_single(&format!("{:?}", self.0)).optional()
    }
}

impl<T> ModuleDisplay for Ignored<T> where T: Sync + Send + core::fmt::Debug + Clone {}

impl<T> Display for Ignored<T>
where
    T: Sync + Send + core::fmt::Debug + Clone,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl<B: AutodiffBackend, T> AutodiffModule<B> for Ignored<T>
where
    B: AutodiffBackend,
    T: Sync + Send + core::fmt::Debug + Clone,
{
    type InnerModule = Ignored<T>;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

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

    use burn_tensor::backend::Backend;
    use burn_tensor::{Device, Tensor};

    use crate::TestBackend;
    use crate::{
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        TestAutodiffBackend,
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
