#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! scalar_float_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        scalar_float_ops!($name, $ops, FloatElem<B>);
    };
    (
        $name:ident,
        $ops:expr,
        $elem:ty
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for $name<D> {
            type Args = ScalarOpsDescription<$elem>;

            fn execute(&self, args: &Self::Args, handles: &mut $crate::HandleContainer<B>) {
                let lhs = handles.get_float_tensor::<D>(&args.lhs);
                let output = $ops(lhs, args.rhs.clone());

                handles.register_float_tensor(&args.out.id, output);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! scalar_float2int_ops {
    (
        $name:ident,
        $ops:expr,
        $elem:ty
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for $name<D> {
            type Args = ScalarOpsDescription<$elem>;

            fn execute(&self, args: &Self::Args, handles: &mut $crate::HandleContainer<B>) {
                let lhs = handles.get_float_tensor::<D>(&args.lhs);
                let output = $ops(lhs, args.rhs.clone());

                handles.register_int_tensor(&args.out.id, output);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! unary_float_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for $name<D> {
            type Args = UnaryOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut $crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D>(&args.input);
                let output = $ops(input);

                handles.register_float_tensor(&args.out.id, output);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! unary_int_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for $name<D> {
            type Args = UnaryOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut $crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D>(&args.input);
                let output = $ops(input);

                handles.register_int_tensor(&args.out.id, output);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! scalar_float_cmp_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for $name<D> {
            type Args = ScalarOpsDescription<FloatElem<B>>;

            fn execute(&self, args: &Self::Args, handles: &mut $crate::HandleContainer<B>) {
                let lhs = handles.get_float_tensor::<D>(&args.lhs);
                let output = $ops(lhs, args.rhs.clone());

                handles.register_bool_tensor(&args.out.id, output);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! scalar_int_cmp_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for $name<D> {
            type Args = ScalarOpsDescription<IntElem<B>>;

            fn execute(&self, args: &Self::Args, handles: &mut $crate::HandleContainer<B>) {
                let lhs = handles.get_int_tensor::<D>(&args.lhs);
                let output = $ops(lhs, args.rhs.clone());

                handles.register_bool_tensor(&args.out.id, output);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! scalar_int_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        scalar_int_ops!($name, $ops, IntElem<B>);
    };
    (
        $name:ident,
        $ops:expr,
        $elem:ty
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for $name<D> {
            type Args = ScalarOpsDescription<$elem>;

            fn execute(&self, args: &Self::Args, handles: &mut $crate::HandleContainer<B>) {
                let lhs = handles.get_int_tensor::<D>(&args.lhs);
                let output = $ops(lhs, args.rhs.clone());

                handles.register_int_tensor(&args.out.id, output);
            }
        }
    };
}
