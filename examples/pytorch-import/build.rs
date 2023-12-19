use burn_import::pytorch::Converter;

fn main() {
    Converter::new()
        .input("pytorch/mnist.pt")
        .out_dir("model/")
        .run_from_script();
}
