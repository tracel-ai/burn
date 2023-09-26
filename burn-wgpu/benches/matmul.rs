use burn_wgpu::benchmark::bench;
use burn_wgpu::WgpuDevice;

fn main() {
    bench(&WgpuDevice::BestAvailable)
}
