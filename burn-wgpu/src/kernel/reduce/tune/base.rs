#[macro_export]
/// Generate an autotune operation for a reduce kernel
macro_rules! reduce_tune_ops {
    ($name:ident, $element_trait:ident, $func:expr) => {
        #[derive(new)]
        pub(crate) struct $name<T: $element_trait, const D: usize> {
            input: WgpuTensor<T, D>,
            output: WgpuTensor<T, D>,
            reduce_dim: usize,
        }

        impl<T: $element_trait, const D: usize> AutotuneOperation for $name<T, D> {
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
