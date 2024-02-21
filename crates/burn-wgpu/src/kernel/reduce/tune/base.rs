#[macro_export]
/// Generate an autotune operation for a reduce kernel
macro_rules! reduce_tune_ops {
    ($name:ident, $func:expr) => {
        #[derive(new)]
        pub(crate) struct $name<R: Runtime, E: JitElement, const D: usize> {
            input: JitTensor<R, E, D>,
            output: JitTensor<R, E, D>,
            reduce_dim: usize,
        }

        impl<R: Runtime, E: JitElement, const D: usize> AutotuneOperation for $name<R, E, D> {
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
