//! Full-level fusion logging and the shared data model used by both the logger and
//! the test-only [`FusionInspector`](crate::inspect::FusionInspector).
//!
//! A single execution of a [`ExecutionStrategy`] is flattened into a sequence of
//! [`FusionBlock`]s — one per strategy leaf — each tagged with a [`BlockKind`]. The
//! logger renders these as a human-readable table; the inspector stores them in
//! `FusionReport`s so tests can assert on what fused together.

use burn_ir::{OperationIr, TensorIr};
use burn_std::config::{fusion::FusionLogLevel, log_fusion};
use core::fmt::Write;

use crate::NumOperations;
use crate::stream::{StreamId, store::ExecutionStrategy};

/// One contiguous run of operations — the output of a single [`ExecutionStrategy`]
/// leaf. A `Composed` strategy flattens into a sequence of these.
#[derive(Debug, Clone)]
pub struct FusionBlock {
    /// Whether this block ran fused or unfused.
    pub kind: BlockKind,
    /// The operations contained in the block, in execution order.
    pub operations: Vec<OperationIr>,
}

/// Whether a [`FusionBlock`] was fused into a single kernel or ran op-by-op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockKind {
    /// The operations were fused into a single kernel produced by the named fuser.
    Fused {
        /// The name of the fuser / optimization (e.g. `"ElementWise"`).
        name: &'static str,
        /// The score the fuser reported for this kernel.
        score: u64,
    },
    /// The operations ran individually; no fusion happened for this block.
    Unfused,
}

/// Emit an execution table for the given `strategy` at [`FusionLogLevel::Full`].
///
/// `global` is the `OperationQueue::global` slice at the time of execution — it holds the
/// real tensor IDs and shapes that the indices inside the strategy's `ordering` fields
/// point into. The table is built lazily: no allocation happens when the log level is
/// below `Full` (except when the inspector is installed, which eagerly captures the
/// structured blocks for test assertions).
pub(crate) fn log_execution_table<O: NumOperations>(
    _id: StreamId,
    strategy: &ExecutionStrategy<O>,
    global: &[OperationIr],
) {
    // When the inspector is active, build the blocks eagerly so we can emit a
    // structured report even when fusion logging is disabled. Otherwise stay lazy.
    #[cfg(feature = "test-util")]
    let blocks_for_inspector: Option<Vec<FusionBlock>> = if crate::inspect::is_installed() {
        let mut blocks = Vec::new();
        collect_blocks(strategy, global, &mut blocks);
        crate::inspect::emit(_id, &blocks);
        Some(blocks)
    } else {
        None
    };

    log_fusion(FusionLogLevel::Full, || {
        #[cfg(feature = "test-util")]
        if let Some(blocks) = &blocks_for_inspector {
            return format_table(blocks);
        }
        let mut blocks = Vec::new();
        collect_blocks(strategy, global, &mut blocks);
        format_table(&blocks)
    });
}

pub(crate) fn collect_blocks<O: NumOperations>(
    strategy: &ExecutionStrategy<O>,
    global: &[OperationIr],
    blocks: &mut Vec<FusionBlock>,
) {
    match strategy {
        ExecutionStrategy::Optimization {
            opt,
            ordering,
            score,
        } => {
            blocks.push(FusionBlock {
                kind: BlockKind::Fused {
                    name: opt.name(),
                    score: *score,
                },
                operations: ordering.iter().map(|&i| global[i].clone()).collect(),
            });
        }
        ExecutionStrategy::Operations { ordering } => {
            blocks.push(FusionBlock {
                kind: BlockKind::Unfused,
                operations: ordering.iter().map(|&i| global[i].clone()).collect(),
            });
        }
        ExecutionStrategy::Composed(items) => {
            for item in items {
                collect_blocks(item, global, blocks);
            }
        }
    }
}

pub(crate) fn format_table(blocks: &[FusionBlock]) -> String {
    let total: usize = blocks.iter().map(|b| b.operations.len()).sum();
    if total == 0 {
        return String::from("fusion block: <empty>");
    }

    // One row per op across all blocks. The `block` column carries the block header
    // on the first row of each block and is left blank for the rest, so each fused
    // block is visually grouped but the name/score appear exactly once.
    struct Row {
        idx: String,
        block: String,
        op: String,
        inputs: String,
        outputs: String,
    }

    let headers = ["idx", "block", "op", "inputs", "outputs"];
    let mut widths = headers.map(str::len);

    let mut rows: Vec<Row> = Vec::with_capacity(total);
    let mut global_idx: usize = 0;
    for block in blocks {
        let header = block_header(&block.kind, block.operations.len());
        for (i, op) in block.operations.iter().enumerate() {
            let row = Row {
                idx: global_idx.to_string(),
                block: if i == 0 {
                    header.clone()
                } else {
                    String::new()
                },
                op: op_kind(op),
                inputs: format_tensors(op.inputs()),
                outputs: format_tensors(op.outputs()),
            };
            widths[0] = widths[0].max(row.idx.len());
            widths[1] = widths[1].max(row.block.len());
            widths[2] = widths[2].max(row.op.len());
            widths[3] = widths[3].max(row.inputs.len());
            widths[4] = widths[4].max(row.outputs.len());
            rows.push(row);
            global_idx += 1;
        }
    }

    // Two-space separators between the 5 columns.
    let table_width = widths.iter().sum::<usize>() + 2 * (widths.len() - 1);

    let mut out = String::new();
    writeln!(out, "{}", "═".repeat(table_width)).unwrap();
    writeln!(
        out,
        "fusion trace ({total} op{}, {} block{})",
        if total == 1 { "" } else { "s" },
        blocks.len(),
        if blocks.len() == 1 { "" } else { "s" },
    )
    .unwrap();

    let write_row = |out: &mut String, cols: [&str; 5]| {
        for (i, c) in cols.iter().enumerate() {
            if i > 0 {
                out.push_str("  ");
            }
            write!(out, "{:<width$}", c, width = widths[i]).unwrap();
        }
        out.push('\n');
    };

    write_row(&mut out, headers);
    let seps = widths.map(|w| "-".repeat(w));
    write_row(
        &mut out,
        [
            seps[0].as_str(),
            seps[1].as_str(),
            seps[2].as_str(),
            seps[3].as_str(),
            seps[4].as_str(),
        ],
    );
    for row in &rows {
        write_row(
            &mut out,
            [&row.idx, &row.block, &row.op, &row.inputs, &row.outputs],
        );
    }

    // Trim the trailing newline so `println!`/`writeln!` sinks don't double up.
    if out.ends_with('\n') {
        out.pop();
    }
    out
}

fn block_header(kind: &BlockKind, size: usize) -> String {
    let ops_suffix = if size == 1 { "op" } else { "ops" };
    match kind {
        BlockKind::Fused { name, score } => {
            format!("▸ fused {name} (score={score}, {size} {ops_suffix})")
        }
        BlockKind::Unfused => format!("▸ un-fused ({size} {ops_suffix})"),
    }
}

/// Take the top-level variant name from a Debug representation (`"Foo(..)"` → `"Foo"`).
fn debug_head(s: &str) -> &str {
    let end = s.find(['(', '{', ' ']).unwrap_or(s.len());
    &s[..end]
}

/// Produce a "Outer::Inner" kind string for every variant that has an inner enum,
/// so the table shows the concrete operation (e.g. `BaseFloat::Reshape`) rather than
/// just the category.
pub(crate) fn op_kind(op: &OperationIr) -> String {
    fn inner(s: String) -> String {
        debug_head(&s).to_string()
    }
    match op {
        OperationIr::BaseFloat(x) => format!("BaseFloat::{}", inner(format!("{x:?}"))),
        OperationIr::BaseInt(x) => format!("BaseInt::{}", inner(format!("{x:?}"))),
        OperationIr::BaseBool(x) => format!("BaseBool::{}", inner(format!("{x:?}"))),
        OperationIr::NumericFloat(_, x) => format!("NumericFloat::{}", inner(format!("{x:?}"))),
        OperationIr::NumericInt(_, x) => format!("NumericInt::{}", inner(format!("{x:?}"))),
        OperationIr::Bool(x) => format!("Bool::{}", inner(format!("{x:?}"))),
        OperationIr::Int(x) => format!("Int::{}", inner(format!("{x:?}"))),
        OperationIr::Float(_, x) => format!("Float::{}", inner(format!("{x:?}"))),
        OperationIr::Module(x) => format!("Module::{}", inner(format!("{x:?}"))),
        OperationIr::Init(x) => format!("Init::{}", inner(format!("{x:?}"))),
        OperationIr::Custom(_) => "Custom".to_string(),
        OperationIr::Drop(_) => "Drop".to_string(),
        #[cfg(feature = "distributed")]
        OperationIr::Distributed(x) => format!("Distributed::{}", inner(format!("{x:?}"))),
    }
}

fn format_tensors<'a, I: Iterator<Item = &'a TensorIr>>(tensors: I) -> String {
    let mut out = String::new();
    let mut first = true;
    for t in tensors {
        if !first {
            out.push_str(", ");
        }
        first = false;
        write!(out, "t{}:", t.id.value()).unwrap();
        write_dtype(&mut out, &t.dtype);
        write_shape(&mut out, &t.shape);
    }
    out
}

fn write_dtype(out: &mut String, dtype: &burn_backend::DType) {
    // Debug gives "F32", "Bool(Native)", etc. Lowercase the initial identifier so the
    // output reads like a Rust type annotation; keep anything after the first paren/brace
    // (e.g. Bool variants or QFloat scheme details) verbatim.
    let dbg = format!("{dtype:?}");
    let split = dbg.find(['(', '{', ' ']).unwrap_or(dbg.len());
    let (head, tail) = dbg.split_at(split);
    for c in head.chars() {
        out.push(c.to_ascii_lowercase());
    }
    out.push_str(tail);
}

fn write_shape(out: &mut String, shape: &burn_backend::Shape) {
    out.push('[');
    let mut first = true;
    for d in shape.iter() {
        if !first {
            out.push(',');
        }
        first = false;
        write!(out, "{d}").unwrap();
    }
    out.push(']');
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{DType, Shape};
    use burn_ir::{BaseOperationIr, TensorId, TensorIr, TensorStatus, UnaryOpIr};

    fn tensor(id: u64, dims: &[usize]) -> TensorIr {
        TensorIr {
            id: TensorId::new(id),
            shape: Shape::new_raw(dims.iter().copied().collect()),
            status: TensorStatus::ReadOnly,
            dtype: DType::F32,
        }
    }

    #[test]
    fn tensor_format_is_compact() {
        let t = tensor(42, &[2, 3, 4]);
        let out = format_tensors(core::iter::once(&t));
        assert_eq!(out, "t42:f32[2,3,4]");
    }

    #[test]
    fn tensor_format_multiple_tensors_comma_separated() {
        let a = tensor(1, &[8]);
        let b = tensor(2, &[]);
        let out = format_tensors([&a, &b].into_iter());
        assert_eq!(out, "t1:f32[8], t2:f32[]");
    }

    #[test]
    fn op_kind_includes_inner_variant() {
        let input = tensor(1, &[4, 2]);
        let out = tensor(2, &[2, 4]);
        let op = OperationIr::BaseFloat(BaseOperationIr::Permute(burn_ir::PermuteOpIr {
            input,
            out,
            axes: vec![1, 0],
        }));
        assert_eq!(op_kind(&op), "BaseFloat::Permute");
    }

    #[test]
    fn op_kind_handles_unary_inside_float() {
        let input = tensor(1, &[4]);
        let out = tensor(2, &[4]);
        let op = OperationIr::Float(
            DType::F32,
            burn_ir::FloatOperationIr::Exp(UnaryOpIr { input, out }),
        );
        assert_eq!(op_kind(&op), "Float::Exp");
    }

    #[test]
    fn format_table_puts_name_and_score_in_block_header() {
        let op = OperationIr::Float(
            DType::F32,
            burn_ir::FloatOperationIr::Exp(UnaryOpIr {
                input: tensor(1, &[4]),
                out: tensor(2, &[4]),
            }),
        );
        let blocks = vec![
            FusionBlock {
                kind: BlockKind::Fused {
                    name: "FusedKernel",
                    score: 42,
                },
                operations: vec![op.clone(), op.clone()],
            },
            FusionBlock {
                kind: BlockKind::Unfused,
                operations: vec![op],
            },
        ];

        let table = format_table(&blocks);

        // Top separator precedes the title line.
        let first_line = table.lines().next().unwrap();
        assert!(
            first_line.chars().all(|c| c == '═'),
            "expected top line of ═, got {first_line:?}"
        );
        assert!(table.contains("\nfusion trace (3 ops, 2 blocks)\n"));
        // Fused header names the optimization and its score exactly once.
        assert!(table.contains("▸ fused FusedKernel (score=42, 2 ops)"));
        // Un-fused header is tagged and sized.
        assert!(table.contains("▸ un-fused (1 op)"));
        // No per-row repetition of "FusedKernel" or "42": they appear exactly once.
        assert_eq!(table.matches("FusedKernel").count(), 1);
        assert_eq!(table.matches("score=42").count(), 1);
        // Indices are global (0, 1, 2 across blocks).
        assert!(table.contains("\n0  "));
        assert!(table.contains("\n1  "));
        assert!(table.contains("\n2  "));
    }
}
