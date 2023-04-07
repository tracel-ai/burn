use crate as burn;

#[macro_export]
macro_rules! constant {
    (module) => {
        type Record = ();

        fn visit<V: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut V) {
            // Nothing to do
        }

        fn map<M: burn::module::ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
            self
        }

        fn load_record(self, _record: Self::Record) -> Self {
            self
        }

        fn into_record(self) -> Self::Record {}
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
