#[macro_export(local_inner_macros)]
macro_rules! binary_float_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusedBackend> Ops<B> for $name<D> {
            type Args = BinaryOpsDescription;

            fn execute(self: Box<Self>, args: Self::Args, handles: &mut crate::HandleContainer<B>) {
                let lhs = handles.get_float_tensor::<D>(&args.lhs);
                let rhs = handles.get_float_tensor(&args.rhs);
                let output = $ops(lhs, rhs);

                handles.register_float_tensor(&args.out.id, output);
            }
        }
    };
}
