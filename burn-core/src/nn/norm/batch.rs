use crate as burn;

use crate::{
    config::Config,
    module::{Module, Param, RunningState},
    tensor::{backend::Backend, Tensor},
};

/// Configuration to create a [BatchNorm](BatchNorm) layer.
#[derive(Config, Debug)]
pub struct BatchNormConfig {
    /// The number of features.
    pub num_features: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    pub epsilon: f64,
    /// Momentum used to update the metrics. Default: 0.1
    #[config(default = 0.1)]
    pub momentum: f64,
}

/// Applies Batch Normalization over a tensor as described in the paper [Batch Normalization](https://arxiv.org/abs/1502.03167)
///
/// `Y = norm(X) * γ + β`
#[derive(Debug)]
pub struct BatchNorm<B: Backend, const D: usize> {
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
    running_mean: RunningState<Tensor<B, 1>>,
    running_var: RunningState<Tensor<B, 1>>,
    momentum: f64,
    epsilon: f64,
}

impl<B: Backend, const D: usize> burn::module::Module<B> for BatchNorm<B, D> {
    type Record = BatchNormRecord<B, D>;
    fn load_record(self, record: Self::Record) -> Self {
        Self {
            gamma: burn::module::Module::<B>::load_record(self.gamma, record.gamma),
            beta: burn::module::Module::<B>::load_record(self.beta, record.beta),
            running_mean: burn::module::Module::<B>::load_record(
                self.running_mean,
                record.running_mean,
            ),
            running_var: burn::module::Module::<B>::load_record(
                self.running_var,
                record.running_var,
            ),
            momentum: burn::module::Module::<B>::load_record(self.momentum, record.momentum),
            epsilon: burn::module::Module::<B>::load_record(self.epsilon, record.epsilon),
        }
    }
    fn into_record(self) -> Self::Record {
        Self::Record {
            gamma: burn::module::Module::<B>::into_record(self.gamma),
            beta: burn::module::Module::<B>::into_record(self.beta),
            running_mean: burn::module::Module::<B>::into_record(self.running_mean),
            running_var: burn::module::Module::<B>::into_record(self.running_var),
            momentum: burn::module::Module::<B>::into_record(self.momentum),
            epsilon: burn::module::Module::<B>::into_record(self.epsilon),
        }
    }
    fn num_params(&self) -> usize {
        let mut num_params = 0;
        num_params += burn::module::Module::<B>::num_params(&self.gamma);
        num_params += burn::module::Module::<B>::num_params(&self.beta);
        num_params += burn::module::Module::<B>::num_params(&self.running_mean);
        num_params += burn::module::Module::<B>::num_params(&self.running_var);
        num_params += burn::module::Module::<B>::num_params(&self.momentum);
        num_params += burn::module::Module::<B>::num_params(&self.epsilon);
        num_params
    }
    fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
        burn::module::Module::visit(&self.gamma, visitor);
        burn::module::Module::visit(&self.beta, visitor);
        burn::module::Module::visit(&self.running_mean, visitor);
        burn::module::Module::visit(&self.running_var, visitor);
        burn::module::Module::visit(&self.momentum, visitor);
        burn::module::Module::visit(&self.epsilon, visitor);
    }
    fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let gamma = burn::module::Module::<B>::map(self.gamma, mapper);
        let beta = burn::module::Module::<B>::map(self.beta, mapper);
        let running_mean = burn::module::Module::<B>::map(self.running_mean, mapper);
        let running_var = burn::module::Module::<B>::map(self.running_var, mapper);
        let momentum = burn::module::Module::<B>::map(self.momentum, mapper);
        let epsilon = burn::module::Module::<B>::map(self.epsilon, mapper);
        Self {
            gamma,
            beta,
            running_mean,
            running_var,
            momentum,
            epsilon,
        }
    }
    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        let devices = burn::module::Module::<B>::collect_devices(&self.gamma, devices);
        let devices = burn::module::Module::<B>::collect_devices(&self.beta, devices);
        let devices = burn::module::Module::<B>::collect_devices(&self.running_mean, devices);
        let devices = burn::module::Module::<B>::collect_devices(&self.running_var, devices);
        let devices = burn::module::Module::<B>::collect_devices(&self.momentum, devices);
        let devices = burn::module::Module::<B>::collect_devices(&self.epsilon, devices);
        devices
    }
    fn to_device(self, device: &B::Device) -> Self {
        let gamma = burn::module::Module::<B>::to_device(self.gamma, device);
        let beta = burn::module::Module::<B>::to_device(self.beta, device);
        let running_mean = burn::module::Module::<B>::to_device(self.running_mean, device);
        let running_var = burn::module::Module::<B>::to_device(self.running_var, device);
        let momentum = burn::module::Module::<B>::to_device(self.momentum, device);
        let epsilon = burn::module::Module::<B>::to_device(self.epsilon, device);
        Self {
            gamma,
            beta,
            running_mean,
            running_var,
            momentum,
            epsilon,
        }
    }
    fn fork(self, device: &B::Device) -> Self {
        let gamma = burn::module::Module::<B>::fork(self.gamma, device);
        let beta = burn::module::Module::<B>::fork(self.beta, device);
        let running_mean = burn::module::Module::<B>::fork(self.running_mean, device);
        let running_var = burn::module::Module::<B>::fork(self.running_var, device);
        let momentum = burn::module::Module::<B>::fork(self.momentum, device);
        let epsilon = burn::module::Module::<B>::fork(self.epsilon, device);
        Self {
            gamma,
            beta,
            running_mean,
            running_var,
            momentum,
            epsilon,
        }
    }
}
impl<B: Backend, const D: usize> burn::module::AutodiffModule<B> for BatchNorm<B, D>
where
    B: burn::tensor::backend::AutodiffBackend,
    <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
{
    type InnerModule = BatchNorm<B::InnerBackend, D>;
    fn valid(&self) -> Self::InnerModule {
        let gamma = burn::module::AutodiffModule::<B>::valid(&self.gamma);
        let beta = burn::module::AutodiffModule::<B>::valid(&self.beta);
        let running_mean = burn::module::AutodiffModule::<B>::valid(&self.running_mean);
        let running_var = burn::module::AutodiffModule::<B>::valid(&self.running_var);
        let momentum = burn::module::AutodiffModule::<B>::valid(&self.momentum);
        let epsilon = burn::module::AutodiffModule::<B>::valid(&self.epsilon);
        Self::InnerModule {
            gamma,
            beta,
            running_mean,
            running_var,
            momentum,
            epsilon,
        }
    }
}
impl<B: Backend, const D: usize> core::fmt::Display for BatchNorm<B, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}[num_params={}]",
            stringify!(BatchNorm),
            self.num_params()
        )
    }
}
impl<B: Backend, const D: usize> Clone for BatchNorm<B, D> {
    fn clone(&self) -> Self {
        let gamma = self.gamma.clone();
        let beta = self.beta.clone();
        let running_mean = self.running_mean.clone();
        let running_var = self.running_var.clone();
        let momentum = self.momentum.clone();
        let epsilon = self.epsilon.clone();
        Self {
            gamma,
            beta,
            running_mean,
            running_var,
            momentum,
            epsilon,
        }
    }
}
#[doc = r" The record type for the module."]
#[derive(burn :: record :: Record, Debug, Clone)]
pub struct BatchNormRecord<B, D> {
    #[doc = r" The module record associative type."]
    pub gamma: <Param<Tensor<B, 1>> as burn::module::Module<B>>::Record,
    #[doc = r" The module record associative type."]
    pub beta: <Param<Tensor<B, 1>> as burn::module::Module<B>>::Record,
    #[doc = r" The module record associative type."]
    pub running_mean: <RunningState<Tensor<B, 1>> as burn::module::Module<B>>::Record,
    #[doc = r" The module record associative type."]
    pub running_var: <RunningState<Tensor<B, 1>> as burn::module::Module<B>>::Record,
    #[doc = r" The module record associative type."]
    pub momentum: <f64 as burn::module::Module<B>>::Record,
    #[doc = r" The module record associative type."]
    pub epsilon: <f64 as burn::module::Module<B>>::Record,
}

impl BatchNormConfig {
    /// Initialize a new [batch norm](BatchNorm) module.
    pub fn init<B: Backend, const D: usize>(&self, device: &B::Device) -> BatchNorm<B, D> {
        let gamma = Tensor::ones([self.num_features], device);
        let beta = Tensor::zeros([self.num_features], device);

        let running_mean = Tensor::zeros([self.num_features], device);
        let running_var = Tensor::ones([self.num_features], device);

        BatchNorm {
            gamma: Param::from(gamma),
            beta: Param::from(beta),
            running_mean: RunningState::new(running_mean),
            running_var: RunningState::new(running_var),
            momentum: self.momentum,
            epsilon: self.epsilon,
        }
    }

    /// Initialize a new [batch norm](BatchNorm) module with a [record](BatchNormRecord).
    pub fn init_with<B: Backend, const D: usize>(
        &self,
        record: BatchNormRecord<B, D>,
    ) -> BatchNorm<B, D> {
        BatchNorm {
            gamma: record.gamma,
            beta: record.beta,
            running_mean: RunningState::from_record(record.running_mean),
            running_var: RunningState::from_record(record.running_var),
            momentum: self.momentum,
            epsilon: self.epsilon,
        }
    }
}

impl<const D: usize, B: Backend> BatchNorm<B, D> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels, ...]`
    /// - output: `[batch_size, channels, ...]`
    pub fn forward<const DI: usize>(&self, input: Tensor<B, DI>) -> Tensor<B, DI> {
        // Should be move to a compilation error when const generic support that kind of
        // validation. https://github.com/rust-lang/rust/issues/76560
        if D + 2 != DI {
            panic!(
                "BatchNorm{}D can only be applied on tensors of size {} with the following shape \
                 [batch_size, channels, ...], received {}D tensor",
                D,
                D + 2,
                DI
            );
        }

        match B::ad_enabled() {
            true => self.forward_train(input),
            false => self.forward_inference(input),
        }
    }

    fn forward_inference<const DI: usize>(&self, input: Tensor<B, DI>) -> Tensor<B, DI> {
        let channels = input.dims()[1];
        let mean = self.running_mean.value();
        let var = self.running_var.value();

        let mut shape = [1; DI];
        shape[1] = channels;

        self.forward_shared(input, mean.reshape(shape), var.reshape(shape))
    }

    fn forward_train<const DI: usize>(&self, input: Tensor<B, DI>) -> Tensor<B, DI> {
        let dims = input.dims();
        let batch_size = dims[0];
        let channels = dims[1];

        let mut shape_unsqueeze = [1; DI];
        let mut flatten_size = batch_size;
        shape_unsqueeze[1] = channels;

        for dim in dims.iter().take(DI).skip(2) {
            flatten_size *= dim;
        }

        let mean = input
            .clone()
            .swap_dims(0, 1)
            .reshape([channels, flatten_size])
            .mean_dim(1)
            .reshape(shape_unsqueeze);

        let var = input
            .clone()
            .sub(mean.clone())
            .powf(2.0)
            .swap_dims(0, 1)
            .reshape([channels, flatten_size])
            .mean_dim(1)
            .reshape(shape_unsqueeze);

        let running_mean = self.running_mean.value_sync();
        let running_var = self.running_var.value_sync();

        let running_mean = running_mean.mul_scalar(1.0 - self.momentum).add(
            mean.clone()
                .detach()
                .mul_scalar(self.momentum)
                .reshape([channels]),
        );
        let running_var = running_var.mul_scalar(1.0 - self.momentum).add(
            var.clone()
                .detach()
                .mul_scalar(self.momentum)
                .reshape([channels]),
        );

        self.running_mean.update(running_mean.detach());
        self.running_var.update(running_var.detach());

        self.forward_shared(input, mean, var)
    }

    fn forward_shared<const DI: usize>(
        &self,
        x: Tensor<B, DI>,
        mean: Tensor<B, DI>,
        var: Tensor<B, DI>,
    ) -> Tensor<B, DI> {
        let channels = x.dims()[1];
        let mut shape = [1; DI];
        shape[1] = channels;

        let std = var.add_scalar(self.epsilon).sqrt();

        let x = x.sub(mean);
        let x = x.div(std);

        let x = x.mul(self.gamma.val().reshape(shape));

        x.add(self.beta.val().reshape(shape))
    }
}

#[cfg(feature = "std")]
#[cfg(test)]
mod tests_1d {
    use super::*;
    use crate::{module::AutodiffModule, TestAutodiffBackend};
    use burn_tensor::Data;

    #[test]
    fn batch_norm_forward_train() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend, 1>(&device);

        let output = module.forward(input_tensor(&device));

        output.to_data().assert_approx_eq(
            &Data::from([
                [
                    [1.1483e+00, 3.7521e-01],
                    [1.6272e-03, 7.5067e-01],
                    [1.6204e+00, -4.5168e-02],
                ],
                [
                    [6.8856e-02, -1.5923e+00],
                    [-1.6318e+00, 8.7949e-01],
                    [-5.3368e-01, -1.0416e+00],
                ],
            ]),
            2,
        );
    }

    #[test]
    fn batch_norm_forward_inference() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend, 1>(&device);

        module.forward(input_tensor(&device));
        let module = module.valid();
        let output = module.forward(input_tensor(&device));

        output.to_data().assert_approx_eq(
            &Data::from([
                [[0.9409, 0.6976], [0.5892, 0.8774], [0.9106, 0.6844]],
                [[0.6012, 0.0782], [-0.0394, 0.9270], [0.6181, 0.5492]],
            ]),
            2,
        );
    }

    fn input_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 3> {
        Tensor::<B, 3>::from_floats(
            [
                [[0.9601, 0.7277], [0.6272, 0.9034], [0.9378, 0.7230]],
                [[0.6356, 0.1362], [0.0249, 0.9509], [0.6600, 0.5945]],
            ],
            device,
        )
    }
}

#[cfg(feature = "std")]
#[cfg(test)]
mod tests_2d {
    use super::*;
    use crate::{module::AutodiffModule, TestAutodiffBackend};
    use burn_tensor::Data;

    #[test]
    fn batch_norm_forward_train() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend, 2>(&device);

        let output = module.forward(input_tensor(&device));

        output.to_data().assert_approx_eq(
            &Data::from([
                [
                    [[1.5136, 0.7506], [-1.2216, 0.1477]],
                    [[0.3135, 1.2252], [-0.4150, 0.6130]],
                    [[1.4186, 0.3372], [-1.5183, 1.5262]],
                ],
                [
                    [[0.4483, -1.1914], [-1.2010, 0.7537]],
                    [[-1.6752, 1.3822], [-0.5058, -0.9381]],
                    [[0.0200, -0.3097], [-0.5715, -0.9026]],
                ],
            ]),
            2,
        );
    }

    #[test]
    fn batch_norm_forward_inference() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend, 2>(&device);

        module.forward(input_tensor(&device));
        let module = module.valid();
        let output = module.forward(input_tensor(&device));

        output.to_data().assert_approx_eq(
            &Data::from([
                [
                    [[0.9538, 0.7103], [0.0808, 0.5179]],
                    [[0.6015, 0.8910], [0.3703, 0.6966]],
                    [[0.9171, 0.6912], [0.3037, 0.9395]],
                ],
                [
                    [[0.6138, 0.0904], [0.0874, 0.7113]],
                    [[-0.0297, 0.9408], [0.3415, 0.2042]],
                    [[0.6250, 0.5561], [0.5013, 0.4323]],
                ],
            ]),
            2,
        );
    }

    #[test]
    fn batch_norm_running_mean() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend, 2>(&device);

        let _output = module.forward(input_tensor(&device));

        let running_mean = module.running_mean.value_sync();

        running_mean
            .reshape([3])
            .into_data()
            .assert_approx_eq(&Data::from([0.0499, 0.0532, 0.0656]), 2);
    }

    #[test]
    fn batch_norm_running_var() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend, 2>(&device);

        let _output = module.forward(input_tensor(&device));

        let running_var = module.running_var.value_sync();

        running_var
            .reshape([3])
            .into_data()
            .assert_approx_eq(&Data::from([0.9106, 0.9105, 0.9045]), 2);
    }

    #[test]
    fn batch_norm_running_mean_inner_module() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend, 2>(&device);

        let _output = module.forward(input_tensor(&device));

        let module_valid = module.valid();
        let running_mean = module_valid.running_mean.value();
        let running_mean_after = module.running_mean.value();

        running_mean_after
            .into_data()
            .assert_approx_eq(&running_mean.into_data(), 3);
    }

    #[test]
    fn batch_norm_grads() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend, 2>(&device);
        let input = input_tensor(&device).require_grad();

        let output = module.forward(input.clone());

        let grads = output.backward();

        module
            .gamma
            .grad(&grads)
            .unwrap()
            .reshape([3])
            .into_data()
            .assert_approx_eq(&Data::from([0.0000e+00, -5.9035e-07, -6.0011e-07]), 3);

        module
            .beta
            .grad(&grads)
            .unwrap()
            .reshape([3])
            .into_data()
            .assert_approx_eq(&Data::from([8., 8., 8.]), 3);

        input.grad(&grads).unwrap().into_data().assert_approx_eq(
            &Data::from([
                [
                    [[0.0000e+00, 0.0000e+00], [0.0000e+00, 0.0000e+00]],
                    [[7.6400e-08, 2.9848e-07], [-1.0110e-07, 1.4933e-07]],
                    [[5.3570e-07, 1.2732e-07], [-5.7336e-07, 5.7632e-07]],
                ],
                [
                    [[0.0000e+00, 0.0000e+00], [0.0000e+00, 0.0000e+00]],
                    [[-4.0807e-07, 3.3673e-07], [-1.2323e-07, -2.2854e-07]],
                    [[7.5642e-09, -1.1695e-07], [-2.1582e-07, -3.4078e-07]],
                ],
            ]),
            4,
        );
    }

    fn input_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 4> {
        Tensor::<B, 4>::from_floats(
            [
                [
                    [[0.9601, 0.7277], [0.1270, 0.5441]],
                    [[0.6272, 0.9034], [0.4066, 0.7179]],
                    [[0.9378, 0.7230], [0.3544, 0.9591]],
                ],
                [
                    [[0.6356, 0.1362], [0.1333, 0.7287]],
                    [[0.0249, 0.9509], [0.3791, 0.2481]],
                    [[0.6600, 0.5945], [0.5424, 0.4767]],
                ],
            ],
            device,
        )
    }
}
