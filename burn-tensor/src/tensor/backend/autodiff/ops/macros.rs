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
            fn partial_left(&self, state: &$crate::graph::ops::BinaryOpsNodeState<T, T, T>) -> T {
                $partial_left(state)
            }

            fn partial_right(&self, state: &$crate::graph::ops::BinaryOpsNodeState<T, T, T>) -> T {
                $partial_right(state)
            }
        }
    };
    (
        ops $ops:ty,
        name $name:ident state $ops_tensor_state:ident,
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
            fn partial(&self, state: &$crate::graph::ops::UnaryOpsNodeState<T, T>) -> T {
                $partial(self.state, state)
            }
        }
    };
    (
        ops $ops:ty,
        name $name:ident,
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
            fn partial(&self, state: &$crate::graph::ops::UnaryRecordedState<T, T>) -> T {
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
        ops $ops:expr,
    ) => {{
        let callback = || {
            let state = $crate::node::NodeState::new_mut($out);

            println!("New binary recorded ops {}", stringify!($ops));
            let ops = $ops;
            let ops = BinaryRecordedOps::new($lhs, $rhs, ops);
            let ops = std::rc::Rc::new(ops);

            let node = $crate::node::Node::new(state, ops);
            std::rc::Rc::new(node)
        };
        callback()
    }};
    (
        input $input:expr,
        out $out:expr,
        ops $ops:expr,
    ) => {{
        let callback = || {
            let state = $crate::node::NodeState::new_mut($out);

            println!("New single recorded ops {}", stringify!($ops));
            let ops = $ops;
            let ops = UnaryRecordedOps::new($input, ops);
            let ops = std::rc::Rc::new(ops);

            let node = $crate::node::Node::new(state, ops);
            std::rc::Rc::new(node)
        };
        callback()
    }};
}
