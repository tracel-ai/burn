#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! scalar_float_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_float_tensor::<B>(&$desc.lhs);
        let output = $ops(lhs, $desc.rhs.elem());

        $handles.register_float_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! scalar_float_dim_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_float_tensor::<B>(&$desc.lhs);
        let output = $ops(lhs, $desc.rhs);

        $handles.register_float_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! reduce_float_dim_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let input = $handles.get_float_tensor::<B>(&$desc.input);
        let output = $ops(input, $desc.axis);

        $handles.register_float_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! reduce_float2int_dim_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let input = $handles.get_float_tensor::<B>(&$desc.input);
        let output = $ops(input, $desc.axis);

        $handles.register_int_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! reduce_int_dim_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let input = $handles.get_int_tensor::<B>(&$desc.input);
        let output = $ops(input, $desc.axis);

        $handles.register_int_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! scalar_float2int_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_float_tensor::<B>(&$desc.lhs);
        let output = $ops(lhs, $desc.rhs);

        $handles.register_int_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! scalar_float_cmp_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_float_tensor::<B>(&$desc.lhs);
        let output = $ops(lhs, $desc.rhs.elem());

        $handles.register_bool_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! unary_float_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_float_tensor::<B>(&$desc.input);
        let output = $ops(lhs);

        $handles.register_float_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! scalar_int_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_int_tensor::<B>(&$desc.lhs);
        let output = $ops(lhs, $desc.rhs.elem());

        $handles.register_int_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! scalar_int_dim_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_int_tensor::<B>(&$desc.lhs);
        let output = $ops(lhs, $desc.rhs);

        $handles.register_int_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! scalar_int_cmp_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_int_tensor::<B>(&$desc.lhs);
        let output = $ops(lhs, $desc.rhs.elem());

        $handles.register_bool_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! unary_int_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_int_tensor::<B>(&$desc.input);
        let output = $ops(lhs);

        $handles.register_int_tensor::<B>(&$desc.out.id, output);
    }};
}
