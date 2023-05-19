#[cfg(test)]
#[cfg(feature = "onnx")]
mod tests {
    use std::fs::read_to_string;
    use std::path::Path;

    use burn::record::FullPrecisionSettings;
    use pretty_assertions::assert_eq;
    use rstest::*;

    fn code<P: AsRef<Path>>(onnx_path: P, record_path: P) -> String {
        let graph = burn_import::onnx::parse_onnx(onnx_path.as_ref());
        let mut graph = graph.into_burn::<FullPrecisionSettings>();
        graph.with_record(
            record_path.as_ref().into(),
            true,
            "burn::record::FullPrecisionSettings",
        );

        burn_import::format_tokens(graph.codegen())
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
        assert_eq!(source_expected, code);
    }
}
