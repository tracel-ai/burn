use burn::tensor::Device;
use simple_regression::{inference, training};

static ARTIFACT_DIR: &str = "/tmp/burn-example-regression";

#[cfg(feature = "flex")]
mod flex {
    use burn::tensor::Device;

    pub fn run() {
        super::run(Device::flex());
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::tensor::{Device, DeviceKind};

    pub fn run() {
        super::run(Device::wgpu(DeviceKind::DefaultDevice));
    }
}

#[cfg(feature = "remote")]
mod remote {
    use burn::{remote::RemoteHost, tensor::Device};

    /// Train on the `server` example's device, at its default address.
    pub fn run() {
        let host = RemoteHost::websocket("ws://localhost:3000");
        let device = Device::remote_options(&host)
            .init()
            .expect("The server can be dialed");
        super::run(device);
    }
}

/// Train a regression model and predict results on a number of samples.
pub fn run(device: Device) {
    training::run(ARTIFACT_DIR, device.clone());
    inference::infer(ARTIFACT_DIR, device)
}

fn main() {
    #[cfg(feature = "flex")]
    flex::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "remote")]
    remote::run();
}
