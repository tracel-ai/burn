use std::collections::LinkedList;

use burn_compute::{
    memory_management::{
        dynamic::{DynamicMemoryManagement, DynamicMemoryManagementOptions},
        MemoryManagement,
    },
    storage::BytesStorage,
};

const MB: usize = 1024 * 1024;

fn main() {
    let start = std::time::Instant::now();
    let storage = BytesStorage::default();
    let mut mm = DynamicMemoryManagement::new(
        storage,
        DynamicMemoryManagementOptions::preset(2048 * MB, 32),
    );
    let mut handles = LinkedList::new();
    for _ in 0..100 * 2048 {
        if handles.len() >= 4000 {
            handles.pop_front();
        }
        let handle = mm.reserve(MB, || {});
        handles.push_back(handle);
    }
    println!("{:?}", start.elapsed());
}
