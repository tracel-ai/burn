#[cfg(all(test, feature = "cpu"))]
mod tests_cpu {
    pub type TestBackend = burn_ndarray::NdArray;

    burn_vision::testgen_all!();
}

#[cfg(all(test, feature = "wgpu"))]
mod tests_wgpu {
    pub type TestBackend = burn_wgpu::Wgpu;

    burn_vision::testgen_all!();
}

#[cfg(all(test, feature = "cuda"))]
mod tests_cuda {
    pub type TestBackend = burn_cuda::Cuda;

    burn_vision::testgen_all!();
}
