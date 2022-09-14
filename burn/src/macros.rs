macro_rules! config {
    ($item:item) => {
        #[derive(new, serde::Serialize, serde::Deserialize, Clone, Debug)]
        $item
    };
}

pub(crate) use config;
