#![allow(clippy::new_without_default)]
#![allow(non_snake_case)]
#[macro_use]
extern crate log;
extern crate android_logger;

use android_logger::Config;
use log::LevelFilter;

pub mod model;

use jni::{
    objects::JByteArray,
    sys::{jint, jobject},
    JNIEnv,
};

use burn::{backend::ndarray::NdArray, tensor::Tensor};

use crate::model::mnist::Model;

#[no_mangle]
pub extern "C" fn Java_com_example_mnistinferenceandroid_MnistInferPageKt_infer(
    env: JNIEnv,
    _: jobject,
    inputImage: JByteArray,
) -> jint {
    // Used to log to the android device's console since regular print statements won't show there, we use the android_logger package
    android_logger::init_once(Config::default().with_max_level(LevelFilter::Trace));

    let input = env
        .convert_byte_array(&inputImage)
        .expect("Error converting byteArray to Int vectors")
        .into_iter()
        .map(f32::from)
        .collect::<Vec<f32>>();

    // Just a POC, usually this would be done in an init function and not in each call
    type Backend = NdArray<f32>;
    let device = <Backend as burn::tensor::backend::Backend>::Device::default();

    let model: Model<Backend> = Model::default();

    // Reshape from the 1D array to 3d tensor [batch, height, width]
    let input =
        Tensor::<Backend, 1>::from_floats(input.as_slice(), &device).reshape([1, 1, 28, 28]);

    // Normalize input: make between [0,1] and make the mean=0 and std=1
    // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
    // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
    let input = ((input / 255) - 0.1307) / 0.3081;

    // Run the tensor input through the model
    let output: Tensor<Backend, 2> = model.forward(input);

    let res = match i32::try_from(output.argmax(1).into_scalar()) {
        Ok(val) => {
            debug!("The number is: {}", val);
            val
        }
        Err(_) => {
            debug!("Model output error!");
            -1
        }
    };
    res
}
