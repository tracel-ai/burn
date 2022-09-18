#[macro_export]
macro_rules! config {
    ($item:item) => {
        #[derive(burn::Config, serde::Serialize, serde::Deserialize, Clone, Debug)]
        $item
    };
}
