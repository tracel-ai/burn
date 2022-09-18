macro_rules! config {
    ($item:item) => {
        #[derive(crate::Config, serde::Serialize, serde::Deserialize, Clone, Debug)]
        $item
    };
}

pub(crate) use config;
