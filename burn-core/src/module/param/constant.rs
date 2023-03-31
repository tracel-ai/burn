use crate as burn;

#[macro_export]
macro_rules! constant {
    (module) => {
        fn devices(&self) -> alloc::vec::Vec<<B as burn_tensor::backend::Backend>::Device> {
            alloc::vec::Vec::new()
        }

        fn to_device(self, _device: &<B as burn_tensor::backend::Backend>::Device) -> Self {
            self
        }

        fn load(
            self,
            _state: &burn::module::State<<B as burn_tensor::backend::Backend>::FloatElem>,
        ) -> Result<Self, burn::module::LoadingError> {
            Ok(self)
        }

        fn state(&self) -> burn::module::State<<B as burn_tensor::backend::Backend>::FloatElem> {
            burn::module::State::StateNamed(burn::module::StateNamed::new())
        }

        fn detach(self) -> Self {
            self
        }

        fn num_params(&self) -> usize {
            0
        }

        fn visit<V: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut V) {
            // Nothing to do
        }

        fn map<M: burn::module::ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
            self
        }
    };

    (ad_module, $type:ty) => {
        type InnerModule = $type;

        fn inner(self) -> Self::InnerModule {
            self
        }

        fn from_inner(module: Self::InnerModule) -> Self {
            module
        }
    };

    ($type:ty) => {
        impl<B: burn::tensor::backend::Backend> burn::module::Module<B> for $type {
            constant!(module);
        }

        impl<B: burn::tensor::backend::ADBackend> burn::module::ADModule<B> for $type {
            constant!(ad_module, $type);
        }
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
