use std::{
    fs::File,
    io::copy,
    path::{Path, PathBuf},
};

/// Download the CSV file from its original source on the web.
/// Panics if the download cannot be completed or the content of the file cannot be written to disk.
pub fn download_csv_if_missing() -> PathBuf {
    // Point file to current example directory
    let example_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
    let file_name = example_dir.join("diabetes.csv");

    if file_name.exists() {
        println!("File already downloaded at {file_name:?}");
    } else {
        // Get file from web
        println!("Downloading file to {file_name:?}");
        let url = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt";
        let mut response = reqwest::blocking::get(url).unwrap();

        // Create file to write the downloaded content to
        let mut file = File::create(&file_name).unwrap();

        // Copy the downloaded contents
        copy(&mut response, &mut file).unwrap();
    };

    file_name
}
