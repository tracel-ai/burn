fn main() {
    if cfg!(target_os = "windows") {
        println!(
            "{}",
            "cargo:warning=The crate is not supported on Windows because of ".to_owned()
                + "Candle's pt bug on Windows "
                + "(see https://github.com/huggingface/candle/issues/1454)."
        );
        std::process::exit(1);
    }
}
