// Declare numerical constraints beside the component; generate only the structural traversal.
macro_rules! impl_optimizer_state {
    ($name:ident {
        $(tensors { $($field:ident),* $(,)? })?
        $(nested { $($nested:ident),* $(,)? })?
        $(optional { $($optional:ident),* $(,)? })?
    }) => {
        impl<const D: usize> crate::StateMappable for $name<D> {
            fn map_tensors<M: crate::OptimizerStateMapper>(mut self, mapper: &mut M) -> Self {
                $($(
                    self.$field = crate::optim::precision::TensorMappable::map(
                        self.$field, mapper, concat!(stringify!($name), ".", stringify!($field)),
                    );
                )*)?
                $($(
                    self.$nested = crate::StateMappable::map_tensors(self.$nested, mapper);
                )*)?
                $($(
                    self.$optional = self.$optional.map(|state|
                        crate::StateMappable::map_tensors(state, mapper)
                    );
                )*)?
                self
            }
        }
    };
}
