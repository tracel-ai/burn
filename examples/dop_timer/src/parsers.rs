use burn::{collective::AllReduceStrategy, tensor::backend::ReduceOperation};
use std::num::ParseIntError;

pub(crate) fn parse_array4(s: &str) -> Result<[usize; 4], String> {
    let parts: Result<Vec<_>, _> = s.split(',').map(|p| p.trim().parse()).collect();
    let parts = parts.map_err(|e: ParseIntError| e.to_string())?;
    parts
        .try_into()
        .map_err(|v: Vec<_>| format!("expected 4 values, got {}", v.len()))
}

pub(crate) fn parse_all_reduce_strategy(s: &str) -> Result<AllReduceStrategy, String> {
    let s = s.trim();
    if let Some(depth) = s.strip_prefix("tree:") {
        let depth = depth.parse::<usize>().map_err(|e| e.to_string())?;
        Ok(AllReduceStrategy::Tree(depth as u32))
    } else if s.eq("centralized") {
        Ok(AllReduceStrategy::Centralized)
    } else if s.eq("ring") {
        Ok(AllReduceStrategy::Ring)
    } else {
        Err(format!("unknown strategy: {}", s))
    }
}

pub(crate) fn parse_reduce_operation(s: &str) -> Result<ReduceOperation, String> {
    let s = s.trim();
    if s.eq("sum") {
        Ok(ReduceOperation::Sum)
    } else if s.eq("mean") {
        Ok(ReduceOperation::Mean)
    } else {
        Err(format!("unknown reduce operation: {}", s))
    }
}
