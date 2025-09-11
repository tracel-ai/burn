#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! scalar_float_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: ScalarOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_float_tensor::<B>(&self.desc.lhs);
                let output = $ops(lhs, self.desc.rhs.elem());

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }
    };
    (
        $name:ident,
        $ops:expr,
        noconvert
    ) => {
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: ScalarOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_float_tensor::<B>(&self.desc.lhs);
                let output = $ops(lhs, self.desc.rhs);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! reduce_float_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: ReduceDimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = $ops(input, self.desc.axis);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! reduce_float2int_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: ReduceDimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = $ops(input, self.desc.axis);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! reduce_int_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: ReduceDimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = $ops(input, self.desc.axis);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
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
    ) => {
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: ScalarOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_float_tensor::<B>(&self.desc.lhs);
                let output = $ops(lhs, self.desc.rhs.clone());

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
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
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = $ops(input);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }
    };
    (
        $name:ident,
        $ops:expr,
        reduce
    ) => {
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = $ops(input);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
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
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = $ops(input);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }
    };
    (
        $name:ident,
        $ops:expr,
        reduce
    ) => {
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = $ops(input);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
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
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: ScalarOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_float_tensor::<B>(&self.desc.lhs);
                let output = $ops(lhs, self.desc.rhs.elem());

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
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
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: ScalarOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_int_tensor::<B>(&self.desc.lhs);
                let output = $ops(lhs, self.desc.rhs.elem());

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
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
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: ScalarOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_int_tensor::<B>(&self.desc.lhs);
                let output = $ops(lhs, self.desc.rhs.elem());

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }
    };
    (
        $name:ident,
        $ops:expr,
        noconvert
    ) => {
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: ScalarOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_int_tensor::<B>(&self.desc.lhs);
                let output = $ops(lhs, self.desc.rhs);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }
    };
}
