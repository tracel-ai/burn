use burn_ndarray::NdArrayDevice;
use text_classification::DbPediaDataset;

type Backend = burn_ndarray::NdArrayBackend<f32>;

fn main() {
    text_classification::inference::infer::<Backend, DbPediaDataset>(
        NdArrayDevice::Cpu,
        "/tmp/text-classification-db-pedia",
        // Samples from the test dataset, but you are free to test with your own text.
        vec![
            " Magnus Eriksson is a Swedish former footballer who played as a forward.".to_string(),
            "Crossbeam Systems is headquartered in Boxborough Massachusetts and has offices in Europe Latin America and Asia Pacific. Crossbeam Systems was acquired by Blue Coat Systems in December 2012 and the Crossbeam brand has been fully absorbed into Blue Coat.".to_string(),
            " Zia is the sequel to the award-winning Island of the Blue Dolphins by Scott O'Dell. It was published in 1976 sixteen years after the publication of the first novel.".to_string(),
        ],
    );
}
