use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::{backend::Backend, Tensor};
use libm::sqrt;

use super::Initializer;

/// Configuration to create a [Linear](Linear) layer.
#[derive(Config, Debug)]
pub struct LinearConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the output features.
    pub d_output: usize,
    /// If a bias should be applied during the linear transformation.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    #[config(default = "Initializer::KaimingUniform{gain:1.0/sqrt(3.0), fan_out_only:false}")]
    pub initializer: Initializer,
}

/// Applies a linear transformation to the input tensor:
///
/// `O = IW + b`
#[derive(Module, Debug)]
pub struct Linear<B: Backend> {
    /// Matrix of shape `[d_input, d_output]` initialized from a uniform distribution:
    ///     `U(-k, k)`, where `k = sqrt(1 / d_input)`
    pub weight: Param<Tensor<B, 2>>,
    /// Vector of size `d_output` initialized from a uniform distribution:
    ///     `U(-k, k)`, where `k = sqrt(1 / d_input)`
    pub bias: Option<Param<Tensor<B, 1>>>,
}

#[derive(Module, Debug)]
struct ModuleWithGenericModule<B: Backend, Mo> {
    module: Mo,
    _backend: core::marker::PhantomData<B>,
}

//impl<B: Backend, Mo> burn::module::Module<B> for ModuleWithGenericModule<B, Mo>
//where
//    Mo: burn::module::Module<B>,
//{
//    type Record = ModuleWithGenericModuleRecord<B, Mo>;
//    fn load_record(self, record: Self::Record) -> Self {
//        Self {
//            module: burn::module::Module::<B>::load_record(self.module, record.module),
//            _backend: burn::module::Module::<B>::load_record(self._backend, record._backend),
//        }
//    }
//    fn into_record(self) -> Self::Record {
//        Self::Record {
//            module: burn::module::Module::<B>::into_record(self.module),
//            _backend: burn::module::Module::<B>::into_record(self._backend),
//        }
//    }
//    fn num_params(&self) -> usize {
//        let mut num_params = 0;
//        num_params += burn::module::Module::<B>::num_params(&self.module);
//        num_params += burn::module::Module::<B>::num_params(&self._backend);
//        num_params
//    }
//    fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
//        burn::module::Module::visit(&self.module, visitor);
//        burn::module::Module::visit(&self._backend, visitor);
//    }
//    fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
//        let module = burn::module::Module::<B>::map(self.module, mapper);
//        let _backend = burn::module::Module::<B>::map(self._backend, mapper);
//        Self { module, _backend }
//    }
//    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
//        let devices = burn::module::Module::<B>::collect_devices(&self.module, devices);
//        let devices = burn::module::Module::<B>::collect_devices(&self._backend, devices);
//        devices
//    }
//    fn to_device(self, device: &B::Device) -> Self {
//        let module = burn::module::Module::<B>::to_device(self.module, device);
//        let _backend = burn::module::Module::<B>::to_device(self._backend, device);
//        Self { module, _backend }
//    }
//    fn fork(self, device: &B::Device) -> Self {
//        let module = burn::module::Module::<B>::fork(self.module, device);
//        let _backend = burn::module::Module::<B>::fork(self._backend, device);
//        Self { module, _backend }
//    }
//}
//impl<B: Backend, Mo> burn::module::AutodiffModule<B> for ModuleWithGenericModule<B, Mo>
//where
//    B: burn::tensor::backend::AutodiffBackend,
//    <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
//    Mo: burn::module::AutodiffModule<B>,
//    <Mo as burn::module::AutodiffModule<B>>::InnerModule: burn::module::Module<B::InnerBackend>,
//{
//    type InnerModule = ModuleWithGenericModule<
//        B::InnerBackend,
//        <Mo as burn::module::AutodiffModule<B>>::InnerModule,
//    >;
//    fn valid(&self) -> Self::InnerModule {
//        let module = burn::module::AutodiffModule::<B>::valid(&self.module);
//        let _backend = burn::module::AutodiffModule::<B>::valid(&self._backend);
//        Self::InnerModule { module, _backend }
//    }
//}
//impl<B: Backend, Mo> core::fmt::Display for ModuleWithGenericModule<B, Mo>
//where
//    Mo: burn::module::Module<B>,
//{
//    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
//        write!(
//            f,
//            "{}[num_params={}]",
//            stringify!(ModuleWithGenericModule),
//            self.num_params()
//        )
//    }
//}
//impl<B: Backend, Mo> Clone for ModuleWithGenericModule<B, Mo>
//where
//    Mo: burn::module::Module<B>,
//{
//    fn clone(&self) -> Self {
//        let module = self.module.clone();
//        let _backend = self._backend.clone();
//        Self { module, _backend }
//    }
//}
//#[doc = r" The record type for the module."]
//pub struct ModuleWithGenericModuleRecord<B: Backend, Mo>
//where
//    Mo: burn::module::Module<B>,
//{
//    #[doc = r" The module record associative type."]
//    pub module: <Mo as burn::module::Module<B>>::Record,
//    #[doc = r" The module record associative type."]
//    pub _backend: <core::marker::PhantomData<B> as burn::module::Module<B>>::Record,
//}
//
//#[doc = r" The record item type for the module."]
//#[derive(burn :: serde :: Serialize, burn :: serde :: Deserialize)]
//#[serde(crate = "burn::serde")]
//#[serde(
//    bound = "< < Mo as burn :: module :: Module < B > > :: Record as burn :: record ::\nRecord > :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned, < < core :: marker :: PhantomData < B > as burn :: module ::\nModule < B > > :: Record as burn :: record :: Record > :: Item < S > : burn ::\nserde :: Serialize + burn :: serde :: de :: DeserializeOwned,"
//)]
//pub struct ModuleWithGenericModuleRecordItem < B : Backend, Mo: Module<B>, S : burn ::
//   record :: PrecisionSettings >
//   {
//       #[doc = r" Field to be serialized."] pub module : < < Mo as burn :: module
//       :: Module < B > > :: Record as burn :: record :: Record > :: Item < S >,
//       #[doc = r" Field to be serialized."] pub _backend : < < core :: marker ::
//       PhantomData < B > as burn :: module :: Module < B > > :: Record as burn ::
//       record :: Record > :: Item < S >,
//   }
//impl<B: Backend, Mo> burn::record::Record for ModuleWithGenericModuleRecord<B, Mo>
//where
//    Mo: burn::module::Module<B>,
//{
//    type Item<S: burn::record::PrecisionSettings> = ModuleWithGenericModuleRecordItem<B, Mo, S>;
//    fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
//        ModuleWithGenericModuleRecordItem {
//            module: burn::record::Record::into_item::<S>(self.module),
//            _backend: burn::record::Record::into_item::<S>(self._backend),
//        }
//    }
//    fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
//        Self {
//            module: burn::record::Record::from_item::<S>(item.module),
//            _backend: burn::record::Record::from_item::<S>(item._backend),
//        }
//    }
//}

impl LinearConfig {
    /// Initialize a new [linear](Linear) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Linear<B> {
        let shape = [self.d_input, self.d_output];
        let weight =
            self.initializer
                .init_with(shape, Some(self.d_input), Some(self.d_output), device);
        let bias = if self.bias {
            Some(self.initializer.init_with(
                [self.d_output],
                Some(self.d_input),
                Some(self.d_output),
                device,
            ))
        } else {
            None
        };

        Linear {
            weight: Param::from(weight),
            bias: bias.map(Param::from),
        }
    }

    /// Initialize a new [linear](Linear) module with a [record](LinearRecord).
    pub fn init_with<B: Backend>(&self, record: LinearRecord<B>) -> Linear<B> {
        Linear {
            weight: record.weight,
            bias: record.bias,
        }
    }
}

impl<B: Backend> Linear<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any, d_input]`
    /// - output: `[..., any, d_output]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let output = input.matmul(self.weight.val().unsqueeze());

        match &self.bias {
            Some(bias) => output + bias.val().unsqueeze(),
            None => output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::{Data, Shape};
    use libm::sqrt;

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = LinearConfig::new(5, 5);
        let k = sqrt(1.0 / config.d_input as f64) as f32;
        let device = Default::default();
        let linear = config.init::<TestBackend>(&device);

        assert_eq!(
            config.initializer,
            Initializer::KaimingUniform {
                gain: 1.0 / sqrt(3.0),
                fan_out_only: false
            }
        );
        linear.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config = LinearConfig::new(5, 5).with_initializer(Initializer::Zeros);
        let device = Default::default();
        let linear = config.init::<TestBackend>(&device);

        assert_eq!(config.initializer, Initializer::Zeros);
        linear
            .weight
            .to_data()
            .assert_approx_eq(&Data::zeros(linear.weight.shape()), 3);
    }

    #[test]
    fn test_linear_forward_no_bias() {
        TestBackend::seed(0);

        let value = 2.;
        let config = LinearConfig::new(2, 3)
            .with_initializer(Initializer::Constant { value })
            .with_bias(false);
        let device = Default::default();
        let linear = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::ones(Shape::new([1, 2]), &device);
        let result = linear.forward(input);
        let expected_result = Tensor::<TestBackend, 2>::from_data([[4., 4., 4.]], &device);

        assert_eq!(result.into_data(), expected_result.into_data());
    }

    #[test]
    fn test_linear_forward_with_bias() {
        TestBackend::seed(0);

        let device = Default::default();

        let value = 2.;
        let config = LinearConfig::new(2, 3).with_initializer(Initializer::Constant { value });
        let linear = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::ones(Shape::new([1, 2]), &device);
        let result = linear.forward(input);
        let expected_result = Tensor::<TestBackend, 2>::from_data([[6., 6., 6.]], &device);

        assert_eq!(result.into_data(), expected_result.into_data());
    }
}
