extern crate dirs;

pub mod source;
pub mod transform;

mod dataset;
pub use dataset::*;

#[cfg(test)]
mod test_data {
    pub fn string_items() -> Vec<String> {
        vec![
            "1 Item".to_string(),
            "2 Items".to_string(),
            "3 Items".to_string(),
            "4 Items".to_string(),
        ]
    }
}
