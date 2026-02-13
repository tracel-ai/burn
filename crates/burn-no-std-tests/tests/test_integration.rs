#![no_std] // Must keep it for testing

use burn_no_std_tests::mlp::*;
use burn_no_std_tests::model::*;

use burn::tensor::{Distribution, Tensor, backend::Backend};
use burn_ndarray::NdArray;

#[test]
fn test_mnist_model_with_random_input() {
    type Backend = NdArray<f32>;

    // Model configurations
    let device = Default::default();
    let mlp_config = MlpConfig::new();
    let mnist_config = MnistConfig::new(mlp_config);
    let mnist_model: Model<Backend> = Model::new(&mnist_config, &device);

    // Pass a fixed seed for random, otherwise a build generated random seed is used
    Backend::seed(&device, mnist_config.seed);

    // Some random input
    let input_shape = [1, 28, 28];
    let input = Tensor::<Backend, 3>::random(input_shape, Distribution::Default, &device);

    // Run through the model
    let output = mnist_model.forward(input);

    assert_eq!(&*output.shape(), [1, 10]);
    assert!(output.to_data().iter::<f32>().all(|x| x <= 1.0));
}
