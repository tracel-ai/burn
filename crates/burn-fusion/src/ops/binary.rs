#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_float_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        #[derive(new)]
        struct $name<B: FusionBackend> {
            desc: BinaryOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_float_tensor::<B>(&self.desc.lhs);
                let rhs = handles.get_float_tensor::<B>(&self.desc.rhs);
                let output = $ops(lhs, rhs);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
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
        struct $name<B: FusionBackend> {
            desc: BinaryOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_float_tensor::<B>(&self.desc.lhs);
                let rhs = handles.get_float_tensor::<B>(&self.desc.rhs);
                let output = $ops(lhs, rhs);

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
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
        struct $name<B: FusionBackend> {
            desc: BinaryOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_int_tensor::<B>(&self.desc.lhs);
                let rhs = handles.get_int_tensor::<B>(&self.desc.rhs);
                let output = $ops(lhs, rhs);

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
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
        struct $name<B: FusionBackend> {
            desc: BinaryOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_int_tensor::<B>(&self.desc.lhs);
                let rhs = handles.get_int_tensor::<B>(&self.desc.rhs);
                let output = $ops(lhs, rhs);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }
    };
}
