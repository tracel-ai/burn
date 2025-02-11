#[cfg(all(test, feature = "test-cpu"))]
mod tests_cpu {
    pub type TestBackend = burn_ndarray::NdArray<f32, i32>;

    burn_vision::testgen_all!();
}

#[cfg(all(test, feature = "test-wgpu"))]
mod tests_wgpu {
    pub type TestBackend = burn_wgpu::Wgpu;

    burn_vision::testgen_all!();
}

#[cfg(all(test, feature = "test-vulkan"))]
mod tests_wgpu {
    pub type TestBackend = burn_wgpu::Vulkan;

    burn_vision::testgen_all!();
}

#[cfg(all(test, feature = "test-cuda"))]
mod tests_cuda {
    pub type TestBackend = burn_cuda::Cuda;

    burn_vision::testgen_all!();
}
