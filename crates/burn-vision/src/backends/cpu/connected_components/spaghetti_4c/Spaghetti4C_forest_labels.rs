/// Workaround for rust-analyzer bug that causes invalid errors on the `include!`.
macro_rules! no_analyze {
    ($tokens:tt) => {
        $tokens
    };
}

pub(crate) use no_analyze;

#[allow(non_snake_case, non_camel_case_types, unused)]
pub enum centerLabels {
    cl_tree_0,
    cl_tree_1,
}

#[allow(non_snake_case, non_camel_case_types, unused)]
pub enum firstLabels {
    fl_tree_0,
    fl_tree_1,
    fl_,
}
