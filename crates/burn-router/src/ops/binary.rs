#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_float_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_float_tensor::<B>(&$desc.lhs);
        let rhs = $handles.get_float_tensor::<B>(&$desc.rhs);
        let output = $ops(lhs, rhs);

        $handles.register_float_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_float_cmp_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_float_tensor::<B>(&$desc.lhs);
        let rhs = $handles.get_float_tensor::<B>(&$desc.rhs);
        let output = $ops(lhs, rhs);

        $handles.register_bool_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_int_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_int_tensor::<B>(&$desc.lhs);
        let rhs = $handles.get_int_tensor::<B>(&$desc.rhs);
        let output = $ops(lhs, rhs);

        $handles.register_int_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_int_cmp_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_int_tensor::<B>(&$desc.lhs);
        let rhs = $handles.get_int_tensor::<B>(&$desc.rhs);
        let output = $ops(lhs, rhs);

        $handles.register_bool_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_bool_ops {
    (
        $handles:expr, $desc:expr, $ops:expr
    ) => {{
        let lhs = $handles.get_bool_tensor::<B>(&$desc.lhs);
        let rhs = $handles.get_bool_tensor::<B>(&$desc.rhs);
        let output = $ops(lhs, rhs);

        $handles.register_bool_tensor::<B>(&$desc.out.id, output);
    }};
}
