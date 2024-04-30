#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_float_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        #[derive(new)]
        struct $name<const D: usize> {
            desc: BinaryOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for $name<D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<<B as ReprBackend>::Handle>) {
                let lhs = handles.get_float_tensor::<B, D>(&self.desc.lhs);
                let rhs = handles.get_float_tensor::<B, D>(&self.desc.rhs);
                let output = $ops(lhs, rhs);

                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
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
        #[derive(new)]
        struct $name<const D: usize> {
            desc: BinaryOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for $name<D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_float_tensor::<B, D>(&self.desc.lhs);
                let rhs = handles.get_float_tensor::<B, D>(&self.desc.rhs);
                let output = $ops(lhs, rhs);

                handles.register_bool_tensor::<B, D>(&self.desc.out.id, output);
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
        #[derive(new)]
        struct $name<const D: usize> {
            desc: BinaryOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for $name<D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_int_tensor::<B, D>(&self.desc.lhs);
                let rhs = handles.get_int_tensor::<B, D>(&self.desc.rhs);
                let output = $ops(lhs, rhs);

                handles.register_bool_tensor::<B, D>(&self.desc.out.id, output);
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
        #[derive(new)]
        struct $name<const D: usize> {
            desc: BinaryOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for $name<D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_int_tensor::<B, D>(&self.desc.lhs);
                let rhs = handles.get_int_tensor::<B, D>(&self.desc.rhs);
                let output = $ops(lhs, rhs);

                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }
    };
}
