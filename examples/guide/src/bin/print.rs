use burn::backend::WebGpu;
use guide::model::ModelConfig;

fn main() {
    type MyBackend = WebGpu<f32, i32>;

    let device = Default::default();
    let model = ModelConfig::new(10, 512).init::<MyBackend>(&device);

    println!("{}", model);
}
