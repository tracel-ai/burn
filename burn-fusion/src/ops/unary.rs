#[macro_export(local_inner_macros)]
macro_rules! scalar_float_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusedBackend> Ops<B> for $name<D> {
            type Args = ScalarOpsDescription<FloatElem<B>>;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let lhs = handles.get_float_tensor::<D>(&args.lhs);
                let output = $ops(lhs, args.rhs.clone());

                handles.register_float_tensor(&args.out.id, output);
            }
        }
    };
}

#[macro_export(local_inner_macros)]
macro_rules! scalar_int_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusedBackend> Ops<B> for $name<D> {
            type Args = ScalarOpsDescription<IntElem<B>>;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let lhs = handles.get_int_tensor::<D>(&args.lhs);
                let output = $ops(lhs, args.rhs.clone());

                handles.register_int_tensor(&args.out.id, output);
            }
        }
    };
}
