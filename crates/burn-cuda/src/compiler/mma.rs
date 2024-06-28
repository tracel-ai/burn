use std::fmt::Display;

use super::{Elem, Variable};

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum FragmentIdent {
    A,
    B,
    Accumulator,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum FragmentLayout {
    ColMajor,
    RowMajor,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct Fragment {
    pub ident: FragmentIdent,
    pub m: u8,
    pub n: u8,
    pub k: u8,
    pub elem: Elem,
    pub layout: Option<FragmentLayout>,
}

/// Warp Matrix-Multiply and Accumulate Instruction.
#[derive(Debug, Clone, Copy)]
pub enum WmmaInstruction {
    /// Fill the fragment with the value.
    Fill { frag: Variable, value: Variable },
    /// Load the value into the fragment given the stride.
    Load {
        frag: Variable,
        value: Variable,
        stride: Variable,
    },
    /// Executes D=A*B+C;
    ///
    /// For implementing a matmul, `D=C` : `C+=A*B`
    Execute {
        frag_a: Variable,
        frag_b: Variable,
        frag_c: Variable,
        frag_d: Variable,
    },
    /// Store the fragment in an output variable following the stride and the layout.
    Store {
        output: Variable,
        frag: Variable,
        stride: Variable,
        layout: FragmentLayout,
    },
}

impl Display for FragmentLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FragmentLayout::ColMajor => f.write_str("wmma::col_major"),
            FragmentLayout::RowMajor => f.write_str("wmma::row_major"),
        }
    }
}

impl Display for FragmentIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FragmentIdent::A => f.write_str("wmma::matrix_a"),
            FragmentIdent::B => f.write_str("wmma::matrix_b"),
            FragmentIdent::Accumulator => f.write_str("wmma::accumulator"),
        }
    }
}

impl Display for Fragment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.layout {
            Some(layout) => f.write_fmt(format_args!(
                "wmma::fragment<{}, {}, {}, {}, {}, {}>",
                self.ident, self.m, self.n, self.k, self.elem, layout
            )),
            None => f.write_fmt(format_args!(
                "wmma::fragment<{}, {}, {}, {}, {}>",
                self.ident, self.m, self.n, self.k, self.elem,
            )),
        }
    }
}

impl Display for WmmaInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WmmaInstruction::Fill { frag, value } => {
                f.write_fmt(format_args!("wmma::fill_fragment({frag}, {value});\n"))
            }
            WmmaInstruction::Load {
                frag,
                value,
                stride,
            } => f.write_fmt(format_args!(
                "wmma::load_matrix_sync({frag}, {value}, {stride});\n"
            )),
            WmmaInstruction::Execute {
                frag_a,
                frag_b,
                frag_c,
                frag_d,
            } => f.write_fmt(format_args!(
                "wmma::mma_sync({frag_d}, {frag_a}, {frag_b}, {frag_c});\n"
            )),
            WmmaInstruction::Store {
                output,
                frag,
                stride,
                layout,
            } => {
                let layout = match layout {
                    FragmentLayout::ColMajor => "wmma::mem_col_major",
                    FragmentLayout::RowMajor => "wmma::mem_row_major",
                };

                f.write_fmt(format_args!(
                    "wmma::store_matrix_sync({output}, {frag}, {stride}, {layout});\n"
                ))
            }
        }
    }
}
