#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_float_ops {
    (
        $ctx:expr, $desc:expr, $ops:expr
    ) => {{
        let handles = &mut $ctx.lock().handles;
        let lhs = handles.get_float_tensor::<B>(&$desc.lhs);
        let rhs = handles.get_float_tensor::<B>(&$desc.rhs);
        let output = $ops(lhs, rhs);

        handles.register_float_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_float_cmp_ops {
    (
        $ctx:expr, $desc:expr, $ops:expr
    ) => {{
        let handles = &mut $ctx.lock().handles;
        let lhs = handles.get_float_tensor::<B>(&$desc.lhs);
        let rhs = handles.get_float_tensor::<B>(&$desc.rhs);
        let output = $ops(lhs, rhs);

        handles.register_bool_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_int_ops {
    (
        $ctx:expr, $desc:expr, $ops:expr
    ) => {{
        let handles = &mut $ctx.lock().handles;
        let lhs = handles.get_int_tensor::<B>(&$desc.lhs);
        let rhs = handles.get_int_tensor::<B>(&$desc.rhs);
        let output = $ops(lhs, rhs);

        handles.register_int_tensor::<B>(&$desc.out.id, output);
    }};
}

#[allow(missing_docs)]
#[macro_export(local_inner_macros)]
macro_rules! binary_int_cmp_ops {
    (
        $ctx:expr, $desc:expr, $ops:expr
    ) => {{
        let handles = &mut $ctx.lock().handles;
        let lhs = handles.get_int_tensor::<B>(&$desc.lhs);
        let rhs = handles.get_int_tensor::<B>(&$desc.rhs);
        let output = $ops(lhs, rhs);

        handles.register_bool_tensor::<B>(&$desc.out.id, output);
    }};
}
