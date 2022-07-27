#[macro_export]
macro_rules! define_ops {
    (
        name $name:ident
    ) => {
        #[derive(Debug)]
        struct $name<P, const D: usize> {
            _kind: $crate::tensor::backend::autodiff::ADKind<P>,
        }

        impl<P: Default, const D: usize> $name<P, D> {
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

        impl<P: Default, const D: usize> $name<P, D> {
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
            P: $crate::tensor::backend::autodiff::ADElement,
            T: $crate::tensor::backend::autodiff::ADCompatibleTensor<P, D>,
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
        $crate::define_ops!(
            name $name,
            state $ops_tensor_state,
        );

        impl<T, P, const D: usize> $ops for $name<P, D>
        where
            P: $crate::tensor::backend::autodiff::ADElement,
            T: $crate::tensor::backend::autodiff::ADCompatibleTensor<P, D>,
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
        $crate::define_ops!(
            name $name
        );

        impl<T, P, const D: usize> $ops for $name<P, D>
        where
            P: $crate::tensor::backend::autodiff::ADElement,
            T: $crate::tensor::backend::autodiff::ADCompatibleTensor<P, D>,
        {
            fn partial(&self, state: &$crate::graph::ops::UnaryOpsNodeState<T, T>) -> T {
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
            let state = $crate::graph::node::ForwardNodeState::new($out);

            let ops = std::sync::Arc::new($ops);
            let ops = $crate::graph::ops::ForwardBinaryRecordedOps::new($lhs, $rhs, ops.clone());
            let ops = std::sync::Arc::new(ops);

            let node = $crate::graph::node::ForwardNode::from_binary(&$lhs, &$rhs, state, ops);
            std::sync::Arc::new(node)
        };
        callback()
    }};
    (
        input $input:expr,
        out $out:expr,
        ops $ops:expr,
    ) => {{
        let callback = || {
            let state = $crate::graph::node::ForwardNodeState::new($out);

            let ops = std::sync::Arc::new($ops);
            let ops = $crate::graph::ops::ForwardUnaryRecordedOps::new($input, ops.clone());
            let ops = std::sync::Arc::new(ops);

            let node = $crate::graph::node::ForwardNode::from_unary(&$input, state, ops);
            std::sync::Arc::new(node)
        };
        callback()
    }};
    (
        init $out:expr
    ) => {{
        let callback = || {
            let state = $crate::graph::node::ForwardNodeState::new($out);

            let ops = $crate::graph::ops::InitRecordedOps::new();
            let ops = std::sync::Arc::new(ops);

            let node = $crate::graph::node::ForwardNode::from_root(state, ops);
            std::sync::Arc::new(node)
        };
        callback()
    }};
}
