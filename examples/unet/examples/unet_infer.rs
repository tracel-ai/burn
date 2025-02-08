use burn::backend::wgpu::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use std::path::Path;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    let training_artifact_dir = Path::new("artifacts");
    let predict_artifact_dir = Path::new("predict");
    let _ = unet::infer::infer::<MyBackend>(
        training_artifact_dir,
        predict_artifact_dir,
        &device,
        Path::new("data").join("images").join("1.png").as_path(),
    );
}
