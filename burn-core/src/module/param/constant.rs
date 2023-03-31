use crate as burn;

#[macro_export]
macro_rules! constant {
    ($type:ty) => {
        impl<B: burn::tensor::backend::Backend> burn::module::Module<B> for $type {
            fn devices(&self) -> Vec<<B as burn_tensor::backend::Backend>::Device> {
                vec![]
            }

            fn to_device(self, device: &<B as burn_tensor::backend::Backend>::Device) -> Self {
                self
            }

            fn load(
                self,
                state: &burn::module::State<<B as burn_tensor::backend::Backend>::FloatElem>,
            ) -> Result<Self, crate::module::LoadingError> {
                Ok(self)
            }

            fn state(
                &self,
            ) -> burn::module::State<<B as burn_tensor::backend::Backend>::FloatElem> {
                burn::module::State::StateNamed(burn::module::StateNamed::new())
            }

            fn detach(self) -> Self {
                self
            }

            fn num_params(&self) -> usize {
                0
            }

            fn visit<V: crate::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
                // Nothing to do
            }

            fn map<M: crate::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
                self
            }
        }

        impl<B: burn::tensor::backend::ADBackend> burn::module::ADModule<B> for $type {
            type InnerModule = $type;

            fn inner(self) -> Self::InnerModule {
                self
            }

            fn from_inner(module: Self::InnerModule) -> Self {
                module
            }
        }
    };
}

constant!(usize);
constant!(f64);
