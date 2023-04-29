#[cfg(test)]
#[cfg(feature = "onnx")]
mod tests {
    use std::fs::read_to_string;
    use std::path::Path;

    use burn_import::onnx::{ModelGen, ModelSourceCode};
    use pretty_assertions::assert_eq;
    use rstest::*;
    use rust_format::Formatter;

    fn code<P: AsRef<Path>>(onnx_path: P, record_path: P) -> String {
        let model = ModelSourceCode::new(onnx_path, record_path);
        let formatter = ModelGen::code_formatter();
        formatter.format_tokens(model.body()).unwrap()
    }

    #[rstest]
    #[case("model1")]
    // #[case("model2")] <- Add more models here
    fn test_codegen(#[case] model_name: &str) {
        let input_file = format!("tests/data/{model_name}/{model_name}.onnx");
        let source_file = format!("tests/data/{model_name}/{model_name}.rs");
        let source_expected: String =
            read_to_string(source_file).expect("Expected source file is missing");
        let code = code(input_file, format!("./{model_name}"));
        assert_eq!(code, source_expected);
    }
}
