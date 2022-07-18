#[macro_export]
macro_rules! define_ops {
    (
        name $name:ident
    ) => {
        #[derive(Debug)]
        struct $name<P, const D: usize> {
            _kind: $crate::tensor::backend::autodiff::ADKind<P>,
        }

        impl<P: Float + Default, const D: usize> $name<P, D> {
            pub fn new() -> Self {
                Self {
                    _kind: $crate::tensor::backend::autodiff::ADKind::new(),
                }
            }
        }
    };
    (
        name $name:ident,
        state $state_ident:ident,
    ) => {
        #[derive(Debug)]
        struct $name<P, const D: usize> {
            pub state: $state_ident,
            _kind: $crate::tensor::backend::autodiff::ADKind<P>,
        }

        impl<P: Float + Default, const D: usize> $name<P, D> {
            pub fn new(value: $state_ident) -> Self {
                Self {
                    state: value,
                    _kind: $crate::tensor::backend::autodiff::ADKind::new(),
                }
            }
        }
    };
}

#[macro_export]
macro_rules! register_ops {
    (
        ops $ops:ty,
        name $name:ident,
        forward $forward:expr,
        partial_left $partial_left:expr,
        partial_right $partial_right:expr,
    ) => {
        $crate::define_ops!(
            name $name
        );

        impl<T, P, const D: usize> $ops for $name<P, D>
        where
            P: $crate::tensor::backend::autodiff::ADFloat,
            T: $crate::tensor::backend::autodiff::ADFloatTensor<P, D>,
        {
            fn forward(&self, left: T, right: T) -> T {
                $forward(left, right)
            }

            fn partial_left(&self, state: &$crate::graph::ops::BinaryRecordedState<T, T, T>) -> T {
                $partial_left(state)
            }

            fn partial_right(&self, state: &$crate::graph::ops::BinaryRecordedState<T, T, T>) -> T {
                $partial_right(state)
            }
        }
    };
    (
        ops $ops:ty,
        name $name:ident state $ops_tensor_state:ident,
        forward $forward:expr,
        partial $partial:expr,
    ) => {
        define_ops!(
            name $name,
            state $ops_tensor_state,
        );

        impl<T, P, const D: usize> $ops for $name<P, D>
        where
            P: $crate::tensor::backend::autodiff::ADFloat,
            T: $crate::tensor::backend::autodiff::ADFloatTensor<P, D>,
        {
            fn forward(&self, input: T) -> T {
                $forward(self.state, input)
            }

            fn partial(&self, state: &$crate::graph::ops::SingleRecordedState<T, T>) -> T {
                $partial(self.state, state)
            }
        }
    };
    (
        ops $ops:ty,
        name $name:ident,
        forward $forward:expr,
        partial $partial:expr,
    ) => {
        define_ops!(
            name $name,
        );

        impl<T, P, const D: usize> $ops for $name<P, D>
        where
            P: $crate::tensor::backend::autodiff::ADFloat,
            T: $crate::tensor::backend::autodiff::ADFloatTensor<P, D>,
        {
            fn forward(&self, input: T) -> T {
                $forward(input)
            }

            fn partial(&self, state: &$crate::graph::ops::SingleRecordedState<T, T>) -> T {
                $partial(state)
            }
        }
    }

}

#[macro_export]
macro_rules! execute_ops {
    (
        lhs $lhs:expr,
        rhs $rhs:expr,
        out $out:expr,
        tape $tape:expr,
        ops $ops:expr,
    ) =>
    {
        {
            let callback = || {
                let node = $crate::node_init!(
                    lhs $lhs,
                    rhs $rhs,
                    out $out,
                );

                let ops = $ops;
                let ops = BinaryRecordedOps::new($lhs, $rhs, node.clone(), ops);
                let ops = Box::new(ops);

                $tape.borrow_mut().add(ops);
                node
            };
            callback()
        }
    };
    (
        input $input:expr,
        out $out:expr,
        tape $tape:expr,
        ops $ops:expr,
    ) =>
    {
        {
            let callback = || {
                let node = $crate::node_init!(
                    input $input,
                    out $out,
                );

                let ops = $ops;
                let ops = SingleRecordedOps::new($input, node.clone(), ops);
                let ops = Box::new(ops);

                $tape.borrow_mut().add(ops);
                node
            };
            callback()
        }
    };
}
