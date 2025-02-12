use burn_ir::{BinaryOpIr, TensorIr};

#[derive(Debug)]
pub enum BinaryOpError {
    #[allow(dead_code)]
    /// Binary op data type mismatch.
    DTypeMismatch {
        lhs: burn_tensor::DType,
        rhs: burn_tensor::DType,
    },
}

// Until we have floating point type promotion, check that lhs and rhs dtypes are the same.
pub(crate) fn check_binary_op(desc: BinaryOpIr) -> Result<BinaryOpIr, BinaryOpError> {
    check_binary_op_types(&desc.lhs, &desc.rhs)?;
    Ok(desc)
}

pub(crate) fn check_binary_op_types(lhs: &TensorIr, rhs: &TensorIr) -> Result<(), BinaryOpError> {
    if lhs.dtype != rhs.dtype {
        Err(BinaryOpError::DTypeMismatch {
            lhs: lhs.dtype,
            rhs: rhs.dtype,
        })
    } else {
        Ok(())
    }
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_float_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        struct $name<B: FusionBackend> {
            desc: BinaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> $name<B> {
            fn new(desc: BinaryOpIr) -> Self {
                Self {
                    desc: $crate::ops::binary::check_binary_op(desc).unwrap(),
                    _b: PhantomData,
                }
            }
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
            desc: BinaryOpIr,
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
        struct $name<B: FusionBackend> {
            desc: BinaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> $name<B> {
            fn new(desc: BinaryOpIr) -> Self {
                Self {
                    desc: $crate::ops::binary::check_binary_op(desc).unwrap(),
                    _b: PhantomData,
                }
            }
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

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_int_ops {
    (
        $name:ident,
        $ops:expr
    ) => {
        #[derive(new)]
        struct $name<B: FusionBackend> {
            desc: BinaryOpIr,
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
