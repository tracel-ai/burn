pub mod devices;
pub mod finetune;
pub mod inference;
pub mod layered;

/// Where `text-classification`'s `ag-news-train` saves the model it trains.
pub const AG_NEWS_ARTIFACTS: &str = "/tmp/text-classification-ag-news";

/// The artifact directory named on the command line, or [`AG_NEWS_ARTIFACTS`].
pub fn artifact_dir() -> String {
    std::env::args()
        .nth(1)
        .unwrap_or_else(|| AG_NEWS_ARTIFACTS.to_string())
}
