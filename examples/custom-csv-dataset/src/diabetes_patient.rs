use serde::{Deserialize, Serialize};

/// Diabetes patient record.
/// For each field, we manually specify the expected header name for serde as all names
/// are capitalized and some field names are not very informative.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DiabetesPatient {
    /// Age in years
    #[serde(rename = "AGE")]
    pub age: i8,

    /// Sex categorical label
    #[serde(rename = "SEX")]
    pub sex: i8,

    /// Body mass index
    #[serde(rename = "BMI")]
    pub bmi: f32,

    /// Average blood pressure
    #[serde(rename = "BP")]
    pub bp: f32,

    /// S1: total serum cholesterol
    #[serde(rename = "S1")]
    pub tc: i16,

    /// S2: low-density lipoproteins
    #[serde(rename = "S2")]
    pub ldl: f32,

    /// S3: high-density lipoproteins
    #[serde(rename = "S3")]
    pub hdl: f32,

    /// S4: total cholesterol
    #[serde(rename = "S4")]
    pub tch: f32,

    /// S5: possibly log of serum triglycerides level
    #[serde(rename = "S5")]
    pub ltg: f32,

    /// S6: blood sugar level
    #[serde(rename = "S6")]
    pub glu: i8,

    /// Y: quantitative measure of disease progression one year after baseline
    #[serde(rename = "Y")]
    pub response: i16,
}
