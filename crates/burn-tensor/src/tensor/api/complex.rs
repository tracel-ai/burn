use burn_std::TensorData;

use crate::{Device, Float, Tensor, kind::ComplexMath};

impl<const D: usize, K> Tensor<D, K>
where
    K: ComplexMath,
{
    /// Returns the complex conjugate of each element.
    ///
    #[cfg_attr(doc, doc = r#"$\mathrm{conj}(a + bi) = a - bi$"#)]
    #[cfg_attr(not(doc), doc = "`conj(a + bi) = a - bi`")]
    ///
    /// # Example
    ///
    /// ```ignore
    /// let device = Default::default();
    /// let tensor = Tensor::<1, Complex>::from_parts([1.0, -2.0], [3.0, 4.0], &device);
    /// let conjugated = tensor.conj();
    /// ```
    pub fn conj(self) -> Self {
        Self::new(K::conj(self.primitive))
    }

    //TODO: Docs and test
    pub fn powc(self, exponent: Self) -> Self {
        Self::new(K::powc(self.primitive, exponent.primitive))
    }

    /// Returns the phase (argument) of each complex element.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \arg\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = arg(x_i)`")]
    ///
    /// The phase is expressed in radians.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let device = Default::default();
    /// let tensor = Tensor::<1, Complex>::from_parts([1.0, 0.0], [0.0, 1.0], &device);
    /// let phase = tensor.phase();
    /// ```
    pub fn phase(self) -> Tensor<D, Float> {
        Tensor::new(K::phase(self.primitive))
    }

    /// Returns the real component of each complex element.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \Re\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = Re(x_i)`")]
    ///
    /// # Example
    ///
    /// ```ignore
    /// let device = Default::default();
    /// let tensor = Tensor::<1, Complex>::from_parts([1.0, -2.0], [3.0, 4.0], &device);
    /// let real = tensor.real();
    /// ```
    pub fn real(self) -> Tensor<D, Float> {
        Tensor::new(K::real(self.primitive))
    }

    /// Returns the imaginary component of each complex element.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \Im\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = Im(x_i)`")]
    ///
    /// # Example
    ///
    /// ```ignore
    /// let device = Default::default();
    /// let tensor = Tensor::<1, Complex>::from_parts([1.0, -2.0], [3.0, 4.0], &device);
    /// let imag = tensor.imag();
    /// ```
    pub fn imag(self) -> Tensor<D, Float> {
        Tensor::new(K::imag(self.primitive))
    }

    /// Returns the magnitude (absolute value) of each complex element.
    ///
    #[cfg_attr(doc, doc = r#"$|a + bi| = \sqrt{a^2 + b^2}$"#)]
    #[cfg_attr(not(doc), doc = "`|a + bi| = sqrt(a^2 + b^2)`")]
    ///
    /// # Example
    ///
    /// ```ignore
    /// let device = Default::default();
    /// let tensor = Tensor::<1, Complex>::from_parts([3.0, 5.0], [4.0, 12.0], &device);
    /// let mag = tensor.magnitude();
    /// ```
    pub fn magnitude(self) -> Tensor<D, Float> {
        Tensor::new(K::magnitude(self.primitive))
    }

    /// Creates a complex tensor from separate real and imaginary host data.
    ///
    /// # Arguments
    ///
    /// * `real` - Host data for the real part.
    /// * `imag` - Host data for the imaginary part.
    /// * `device` - The device where the resulting tensor is allocated.
    ///
    /// # Returns
    ///
    /// A complex tensor where each element is formed from the provided parts.
    ///
    #[cfg_attr(doc, doc = r#"$z_i = \mathrm{real}_i + \mathrm{imag}_i\, i$"#)]
    #[cfg_attr(not(doc), doc = "`z_i = real_i + imag_i * i`")]
    ///
    /// # Example
    ///
    /// ```ignore
    /// let device = Default::default();
    /// let tensor = Tensor::<1, Complex>::from_parts([1.0, -2.0], [3.0, 4.0], &device);
    /// ```
    pub fn from_parts<T>(real: T, imag: T, device: &Device) -> Self
    where
        T: Into<TensorData>,
    {
        Self::new(K::from_parts(real.into(), imag.into(), device))
    }

    /// Creates a complex tensor from interleaved host data.
    ///
    /// The input data is expected to store real and imaginary values in alternating
    /// order (interleaved layout).
    ///
    /// # Arguments
    ///
    /// * `data` - Interleaved complex tensor data.
    /// * `device` - The device where the resulting tensor is allocated.
    ///
    /// # Returns
    ///
    /// A complex tensor backed by `data` on the target `device`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let device = Default::default();
    /// let data = TensorData::from([1.0_f32, 3.0, -2.0, 4.0]);
    /// let tensor = Tensor::<1, Complex>::from_interleaved_data(data, &device);
    /// ```
    pub fn from_interleaved_data(data: TensorData, device: &Device) -> Self {
        Self::new(K::from_interleaved_data(data, device))
    }

    /// Creates a complex tensor from magnitude and phase (polar form).
    ///
    /// # Arguments
    ///
    /// * `magnitude` - Tensor containing magnitudes (`|z|`).
    /// * `phase` - Tensor containing phases (`arg(z)`) in radians.
    ///
    /// # Returns
    ///
    /// A complex tensor constructed from polar coordinates.
    ///
    #[cfg_attr(
        doc,
        doc = r#"$z_i = r_i\left(\cos\left(\theta_i\right) + i\sin\left(\theta_i\right)\right)$"#
    )]
    #[cfg_attr(not(doc), doc = "`z_i = r_i * (cos(theta_i) + i * sin(theta_i))`")]
    ///
    /// # Example
    ///
    /// ```ignore
    /// let device = Default::default();
    /// let r = Tensor::<1>::from_data([1.0, 2.0], &device);
    /// let theta = Tensor::<1>::from_data([0.0, core::f32::consts::FRAC_PI_2], &device);
    /// let tensor = Tensor::<1, Complex>::from_polar(r, theta);
    /// ```
    pub fn from_polar(magnitude: Tensor<D, Float>, phase: Tensor<D, Float>) -> Self {
        Self::new(K::from_polar(magnitude.primitive, phase.primitive))
    }
}
