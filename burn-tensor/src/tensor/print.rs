use crate::tensor::Data;

impl<P: std::fmt::Debug, const D: usize> std::fmt::Display for Data<P, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("{:?}", &self.value).as_str())
    }
}
