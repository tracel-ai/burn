use burn_std::TensorData;

use crate::{Complex, Device, Float, Tensor, TensorCreationOptions, kind::ComplexMath};

impl<const D: usize> Tensor<D, Complex> {
    /// Creates a complex tensor from interleaved host data.
    pub fn from_complex<A: Into<TensorData>>(complexes: A, device: &Device) -> Self {
        let out_dtype = device.settings().complex_dtype;
        Self::from_data(
            complexes.into(),
            TensorCreationOptions::new(device.clone()).with_dtype(out_dtype.into()),
        )
    }
}
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

    /// Applies element wise power operation with a complex Tensor exponent.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to apply the power operation with.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor,ComplexScalar, Shape, Int};
    ///
    /// fn example() {
    ///    let device = Default::default();
    ///    let tensor1 = Tensor::<2, Complex>::from_ints([[ComplexScalar::new(1.0, -2.0), ComplexScalar::new(3.0, 4.0), ComplexScalar::new(0.0, -1.0)], [ComplexScalar::new(1.0, -2.0), ComplexScalar::new(0.0, -1.0), ComplexScalar::new(2.0, 2.0)]], &device);
    ///    let tensor2 = Tensor::<2, Complex>::from_complex([[ComplexScalar::new(5.0, -1.0), ComplexScalar::new(2.0, 3.0), ComplexScalar::new(1.0, -2.0)], [ComplexScalar::new(1.0, -3.0), ComplexScalar::new(1.0, -3.0), ComplexScalar::new(6.0, 2.0)]], &device);
    ///    let tensor = tensor1.powc(tensor2);
    ///    println!("{tensor}");
    ///    // [[ 1.84452120e+01-1.05764765e+00i,  1.42600948e+00+6.02434630e-01i,
    ///    // 2.64608933e-18-4.32139183e-02i],
    ///    //  [-7.49735280e-02+2.99204278e-02i,  5.50067930e-19-8.98329102e-03i,
    ///    //  9.29602961e+01+5.18329310e+01i]]
    /// }
    /// ```
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

    /// Create a Complex Tensor from a float tensor representing the real part, filling the imaginary part with zeros.
    pub fn from_real<T>(real: T, device: &Device) -> Self
    where
        T: Into<TensorData>,
    {
        Self::new(K::from_real(real.into(), device))
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

//for some reason, implementing sub causes a compilation error originating from order
// saying there are multiple implementations satisfying the constraints.
// // Complex Tensor + Float Tensor
// impl<const D: usize> core::ops::Add<Tensor<D, Float>> for Tensor<D, Complex> {
//     type Output = Self;

//     fn add(self, rhs: Tensor<D, Float>) -> Self::Output {
//         let device = self.device();
//         self + Tensor::<D, Complex>::from_real(rhs.into_data(), &device)
//     }
// }

// // Complex Tensor + Float Tensor
// impl<const D: usize> core::ops::Sub<Tensor<D, Float>> for Tensor<D, Complex> {
//     type Output = Self;

//     fn sub(self, rhs: Tensor<D, Float>) -> Self::Output {
//         let device = self.device();
//         self - Tensor::<D, Complex>::from_real(rhs.into_data(), &device)
//     }
// }
