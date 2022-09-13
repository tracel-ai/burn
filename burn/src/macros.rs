macro_rules! config {
    ($item:item) => {
        #[derive(new, serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq, Eq)]
        $item
    };
}

pub(crate) use config;
