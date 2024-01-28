use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use reqwest::Client;
use std::cmp::min;
use std::io::Write;

/// Download the file at the specified url to a bytes vector.
/// File download progress is reported with the help of a [progress bar](indicatif).
#[tokio::main(flavor = "current_thread")]
pub async fn download_file_as_bytes(url: &str, message: &str) -> Vec<u8> {
    // Get file from web
    let mut response = Client::new().get(url).send().await.unwrap();
    let total_size = response.content_length().unwrap();

    // Pretty progress bar
    let pb = ProgressBar::new(total_size);
    let msg = message.to_owned();
    pb.set_style(
        ProgressStyle::with_template(
            "{msg}\n    {wide_bar:.cyan/blue} {bytes}/{total_bytes} ({eta})",
        )
        .unwrap()
        .with_key(
            "eta",
            |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
            },
        )
        .progress_chars("▬  "),
    );
    pb.set_message(msg.clone());

    // Read stream into bytes
    let mut downloaded: u64 = 0;
    let mut bytes: Vec<u8> = Vec::with_capacity(total_size as usize);
    while let Some(chunk) = response.chunk().await.unwrap() {
        let num_bytes = bytes.write(&chunk).unwrap();
        let new = min(downloaded + (num_bytes as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }
    pb.finish_with_message(msg);

    bytes
}
