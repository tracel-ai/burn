//! Full-level fusion logging: builds a table of every operation that was executed as
//! part of a block optimization, tagging each row as fused, out-of-order, or part of a
//! composed strategy.

use burn_ir::{OperationIr, TensorIr};
use burn_std::config::{fusion::FusionLogLevel, log_fusion};
use core::fmt::Write;
use std::any::type_name_of_val;

use crate::NumOperations;
use crate::stream::store::ExecutionStrategy;

/// Emit an execution table for the given `strategy` at [`FusionLogLevel::Full`].
///
/// `global` is the `OperationQueue::global` slice at the time of execution — it holds the
/// real tensor IDs and shapes that the indices inside the strategy's `ordering` fields
/// point into. The table is built lazily: no allocation happens when the log level is
/// below `Full`.
pub(crate) fn log_execution_table<O: NumOperations>(
    strategy: &ExecutionStrategy<O>,
    global: &[OperationIr],
) {
    log_fusion(FusionLogLevel::Full, || {
        let mut entries = Vec::new();
        collect_entries(strategy, global, &mut entries);
        format_table(&entries)
    });
}

struct Entry {
    kind: EntryKind,
    operation: OperationIr,
}

enum EntryKind {
    /// Part of a fused optimization. Carries the Rust type name of the optimization and
    /// the position of this op within the fused block.
    Fused {
        opt_type: &'static str,
        index_in_block: usize,
        block_size: usize,
    },
    /// Executed as a plain operation, possibly out of registration order.
    OutOfOrder,
}

fn collect_entries<O: NumOperations>(
    strategy: &ExecutionStrategy<O>,
    global: &[OperationIr],
    entries: &mut Vec<Entry>,
) {
    match strategy {
        ExecutionStrategy::Optimization { opt, ordering } => {
            let opt_type = short_type_name(type_name_of_val(opt));
            let block_size = ordering.len();
            for (i, &idx) in ordering.iter().enumerate() {
                entries.push(Entry {
                    kind: EntryKind::Fused {
                        opt_type,
                        index_in_block: i,
                        block_size,
                    },
                    operation: global[idx].clone(),
                });
            }
        }
        ExecutionStrategy::Operations { ordering } => {
            for &idx in ordering.iter() {
                entries.push(Entry {
                    kind: EntryKind::OutOfOrder,
                    operation: global[idx].clone(),
                });
            }
        }
        ExecutionStrategy::Composed(items) => {
            for item in items {
                collect_entries(item, global, entries);
            }
        }
    }
}

fn format_table(entries: &[Entry]) -> String {
    if entries.is_empty() {
        return String::from("fusion execution: <empty>");
    }

    struct Row {
        order: String,
        kind: String,
        op: String,
        inputs: String,
        outputs: String,
    }

    let rows: Vec<Row> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| Row {
            order: i.to_string(),
            kind: match &e.kind {
                EntryKind::Fused {
                    opt_type,
                    index_in_block,
                    block_size,
                } => format!("fused [{index_in_block}/{block_size}] {opt_type}"),
                EntryKind::OutOfOrder => "out-of-order".to_string(),
            },
            op: op_kind(&e.operation),
            inputs: format_tensors(e.operation.inputs()),
            outputs: format_tensors(e.operation.outputs()),
        })
        .collect();

    let headers = ["idx", "kind", "op", "inputs", "outputs"];
    let mut widths = [
        headers[0].len(),
        headers[1].len(),
        headers[2].len(),
        headers[3].len(),
        headers[4].len(),
    ];
    for row in &rows {
        widths[0] = widths[0].max(row.order.len());
        widths[1] = widths[1].max(row.kind.len());
        widths[2] = widths[2].max(row.op.len());
        widths[3] = widths[3].max(row.inputs.len());
        widths[4] = widths[4].max(row.outputs.len());
    }

    let mut out = String::new();
    writeln!(
        out,
        "fusion execution ({} op{})",
        entries.len(),
        if entries.len() == 1 { "" } else { "s" }
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
    let seps: [String; 5] = [
        "-".repeat(widths[0]),
        "-".repeat(widths[1]),
        "-".repeat(widths[2]),
        "-".repeat(widths[3]),
        "-".repeat(widths[4]),
    ];
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
            [&row.order, &row.kind, &row.op, &row.inputs, &row.outputs],
        );
    }
    // Trim the trailing newline so `println!`/`writeln!` sinks don't double up.
    if out.ends_with('\n') {
        out.pop();
    }
    out
}

/// Extract a short, single-segment type name: `"my::crate::path::Kernel<T>"` → `"Kernel"`.
fn short_type_name(full: &'static str) -> &'static str {
    let before_generics = match full.find('<') {
        Some(i) => &full[..i],
        None => full,
    };
    before_generics.rsplit("::").next().unwrap_or(full)
}

/// Take only the top-level variant name from a Debug representation. For nested enums
/// like `Float(F32, Exp(UnaryOpIr { .. }))`, we also peek at the inner variant.
fn op_kind(op: &OperationIr) -> String {
    let top = debug_head(&format!("{op:?}"));
    match op {
        OperationIr::NumericFloat(_, inner) => format!("NumericFloat::{}", debug_head(&format!("{inner:?}"))),
        OperationIr::NumericInt(_, inner) => format!("NumericInt::{}", debug_head(&format!("{inner:?}"))),
        OperationIr::Float(_, inner) => format!("Float::{}", debug_head(&format!("{inner:?}"))),
        _ => top,
    }
}

fn debug_head(s: &str) -> String {
    let end = s
        .find(|c: char| matches!(c, '(' | '{' | ' '))
        .unwrap_or(s.len());
    s[..end].to_string()
}

fn format_tensors<'a, I: Iterator<Item = &'a TensorIr>>(tensors: I) -> String {
    let mut out = String::new();
    let mut first = true;
    for t in tensors {
        if !first {
            out.push_str(", ");
        }
        first = false;
        write!(out, "{:?}{}#{}", t.dtype, t.shape, t.id).unwrap();
    }
    out
}
