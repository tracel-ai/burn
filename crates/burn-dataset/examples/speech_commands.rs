#[cfg(feature = "audio")]
use burn_dataset::{audio::SpeechCommandsDataset, Dataset};

#[cfg(feature = "audio")]
fn speech_command() {
    let index: usize = 4835;
    let test = SpeechCommandsDataset::test();
    let item = test.get(index).unwrap();

    println!("Item: {:?}", item);
    println!("Item Length: {:?}", item.audio_samples.len());
    println!("Label: {}", item.label.to_string());

    assert_eq!(test.len(), 4890);
    assert_eq!(item.label.to_string(), "Yes");
    assert_eq!(item.sample_rate, 16000);
    assert_eq!(item.audio_samples.len(), 16000);
}

fn main() {
    #[cfg(feature = "audio")]
    speech_command()
}
