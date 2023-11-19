#[macro_export]
/// Generate an autotune operation for a reduce kernel
macro_rules! reduce_tune_ops {
    ($name:ident, $func:expr) => {
        #[derive(new)]
        pub(crate) struct $name<E: WgpuElement, const D: usize> {
            input: WgpuTensor<E, D>,
            output: WgpuTensor<E, D>,
            reduce_dim: usize,
        }

        impl<E: WgpuElement, const D: usize> AutotuneOperation for $name<E, D> {
            fn execute(self: Box<Self>) {
                #[allow(clippy::redundant_closure_call)]
                $func(self.input, self.output, self.reduce_dim);
            }

            fn clone(&self) -> Box<dyn AutotuneOperation> {
                Box::new(Self {
                    input: self.input.clone(),
                    output: self.output.clone(),
                    reduce_dim: self.reduce_dim.clone(),
                })
            }
        }
    };
}
