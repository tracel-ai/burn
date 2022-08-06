#[macro_export(local_inner_macros)]
macro_rules! define_ops {
    (
        name $name:ident
    ) => {
        #[derive(Debug)]
        struct $name<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize> $name<B, D> {
            pub fn new() -> Self {
                Self { _b: B::default() }
            }
        }
    };
    (
        name $name:ident,
        state $state_ident:ty,
    ) => {
        #[derive(Debug)]
        struct $name<B: Backend, const D: usize> {
            pub state: $state_ident,
            _b: B,
        }

        impl<B: Backend, const D: usize> $name<B, D> {
            pub fn new(value: $state_ident) -> Self {
                Self {
                    state: value,
                    _b: B::default(),
                }
            }
        }
    };
}

#[macro_export(local_inner_macros)]
macro_rules! register_ops {
    (
        ops $ops:ident,
        name $name:ident,
        partial_left $partial_left:expr,
        partial_right $partial_right:expr,
    ) => {
        $crate::define_ops!(
            name $name
        );

        impl<B: Backend, const D: usize> $ops<B::TensorPrimitive<D>, B::TensorPrimitive<D>, B::TensorPrimitive<D>> for $name<B, D>
        {
            fn partial_left(
                &self,
                state: &$crate::graph::ops::BinaryOpsNodeState<
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>
                >
            ) -> B::TensorPrimitive<D> {
                $partial_left(state)
            }
            fn partial_right(
                &self,
                state: &$crate::graph::ops::BinaryOpsNodeState<
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>
                >
            ) -> B::TensorPrimitive<D> {

                $partial_right(state)
            }
        }
    };
    (
        ops $ops:ident,
        name $name:ident state $ops_tensor_state:ty,
        partial $partial:expr,
    ) => {
        $crate::define_ops!(
            name $name,
            state $ops_tensor_state,
        );

        impl<B: Backend, const D: usize> $ops<B::TensorPrimitive<D>, B::TensorPrimitive<D>> for $name<B, D>
        {
            fn partial(&self, state: &$crate::graph::ops::UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>) -> B::TensorPrimitive<D> {
                $partial(self.state, state)
            }
        }
    };
    (
        ops $ops:ident,
        name $name:ident,
        partial $partial:expr,
    ) => {
        $crate::define_ops!(
            name $name
        );

        impl<B: Backend, const D: usize> $ops<B::TensorPrimitive<D>, B::TensorPrimitive<D>> for $name<B, D>
        {
            fn partial(&self, state: &$crate::graph::ops::UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>) -> B::TensorPrimitive<D> {
                $partial(state)
            }
        }
    }

}

#[macro_export(local_inner_macros)]
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
