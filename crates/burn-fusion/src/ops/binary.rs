#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_float_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        #[derive(Debug)]
        struct $name<B: FusionBackend> {
            desc: BinaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> $name<B> {
            fn new(desc: BinaryOpIr) -> Self {
                Self {
                    desc,
                    _b: PhantomData,
                }
            }
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
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
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: BinaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
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
        #[derive(Debug)]
        struct $name<B: FusionBackend> {
            desc: BinaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> $name<B> {
            fn new(desc: BinaryOpIr) -> Self {
                Self {
                    desc,
                    _b: PhantomData,
                }
            }
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_int_tensor::<B>(&self.desc.lhs);
                let rhs = handles.get_int_tensor::<B>(&self.desc.rhs);
                let output = $ops(lhs, rhs);

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_int_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: BinaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_int_tensor::<B>(&self.desc.lhs);
                let rhs = handles.get_int_tensor::<B>(&self.desc.rhs);
                let output = $ops(lhs, rhs);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }
    };
}
