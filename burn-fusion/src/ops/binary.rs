#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_float_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for $name<D> {
            type Args = BinaryOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut $crate::HandleContainer<B>) {
                let lhs = handles.get_float_tensor::<D>(&args.lhs);
                let rhs = handles.get_float_tensor(&args.rhs);
                let output = $ops(lhs, rhs);

                handles.register_float_tensor(&args.out.id, output);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_float_cmp_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for $name<D> {
            type Args = BinaryOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut $crate::HandleContainer<B>) {
                let lhs = handles.get_float_tensor::<D>(&args.lhs);
                let rhs = handles.get_float_tensor(&args.rhs);
                let output = $ops(lhs, rhs);

                handles.register_bool_tensor(&args.out.id, output);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_int_cmp_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for $name<D> {
            type Args = BinaryOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut $crate::HandleContainer<B>) {
                let lhs = handles.get_int_tensor::<D>(&args.lhs);
                let rhs = handles.get_int_tensor(&args.rhs);
                let output = $ops(lhs, rhs);

                handles.register_bool_tensor(&args.out.id, output);
            }
        }
    };
}

pub(crate) fn binary_ops_shape(lhs: &[usize], rhs: &[usize]) -> Vec<usize> {
    let mut shape_out = Vec::with_capacity(lhs.len());

    for (l, r) in lhs.iter().zip(rhs.iter()) {
        shape_out.push(usize::max(*l, *r));
    }

    shape_out
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_int_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        struct $name<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for $name<D> {
            type Args = BinaryOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut $crate::HandleContainer<B>) {
                let lhs = handles.get_int_tensor::<D>(&args.lhs);
                let rhs = handles.get_int_tensor(&args.rhs);
                let output = $ops(lhs, rhs);

                handles.register_int_tensor(&args.out.id, output);
            }
        }
    };
}
