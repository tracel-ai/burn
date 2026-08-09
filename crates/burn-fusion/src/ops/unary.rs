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
                let output = $ops(lhs, self.desc.rhs.into());

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
macro_rules! reduce_ops {
    ($name:ident, float, $ops:expr) => {
        $crate::reduce_ops!(@impl $name, ReduceDimOpIr, get_float_tensor, register_float_tensor, |input, desc| $ops(input, desc.axis, desc.accumulator_len));
    };
    ($name:ident, float => int, $ops:expr) => {
        $crate::reduce_ops!(@impl $name, ReduceDimOpIr, get_float_tensor, register_int_tensor, |input, desc| $ops(input, desc.axis, desc.accumulator_len, desc.out.dtype.into()));
    };
    ($name:ident, int, $ops:expr) => {
        $crate::reduce_ops!(@impl $name, ReduceDimOpIr, get_int_tensor, register_int_tensor, |input, desc| $ops(input, desc.axis, desc.accumulator_len));
    };
    ($name:ident, bool, $ops:expr) => {
        $crate::reduce_ops!(@impl $name, ReduceDimOpIr, get_bool_tensor, register_bool_tensor, |input, desc| $ops(input, desc.axis));
    };
    ($name:ident, float => bool, $ops:expr) => {
        $crate::reduce_ops!(@impl $name, ReduceDimOpIr, get_float_tensor, register_bool_tensor, |input, desc| $ops(input, desc.axis, desc.out.dtype.into()));
    };
    ($name:ident, int => bool, $ops:expr) => {
        $crate::reduce_ops!(@impl $name, ReduceDimOpIr, get_int_tensor, register_bool_tensor, |input, desc| $ops(input, desc.axis, desc.out.dtype.into()));
    };
    ($name:ident, bool, whole, $ops:expr) => {
        $crate::reduce_ops!(@impl $name, ReduceOpIr, get_bool_tensor, register_bool_tensor, |input, _desc| $ops(input));
    };
    ($name:ident, float => bool, whole, $ops:expr) => {
        $crate::reduce_ops!(@impl $name, ReduceOpIr, get_float_tensor, register_bool_tensor, |input, desc| $ops(input, desc.out.dtype.into()));
    };
    ($name:ident, int => bool, whole, $ops:expr) => {
        $crate::reduce_ops!(@impl $name, ReduceOpIr, get_int_tensor, register_bool_tensor, |input, desc| $ops(input, desc.out.dtype.into()));
    };
    (
        @impl
        $name:ident,
        $desc:ty,
        $get:ident,
        $register:ident,
        |$input:ident, $ir:ident| $ops:expr
    ) => {
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: $desc,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let $input = handles.$get::<B>(&self.desc.input);
                let $ir = &self.desc;
                let output = $ops;

                handles.$register::<B>(&self.desc.out.id, output);
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
                let output = $ops(lhs, self.desc.rhs.into(), self.desc.out.dtype.into());

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
                let output = $ops(lhs, self.desc.rhs.into(), self.desc.out.dtype.into());

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
                let output = $ops(lhs, self.desc.rhs.into());

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
