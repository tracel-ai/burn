//#[burn_tensor_testgen::testgen(complex)]
mod common;

use burn_complex::utils::interleaved_data_to_raw_float_data;

use burn_tensor::Tolerance;
use burn_tensor::{Complex, TensorData};
use common::*;

use burn_complex::kind::ComplexOnlyOps;

#[test]
fn test_complex_conj() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            Complex::<f32> {
                real: 1.0,
                imag: -2.0,
            },
            Complex::<f32> {
                real: -3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let result = tensor.conj();
    let data = result.into_data();

    let expected = TensorData::from([[
        Complex::<f32> {
            real: 1.0,
            imag: 2.0,
        }, // conjugate flips sign of imaginary part
        Complex::<f32> {
            real: -3.0,
            imag: -4.0,
        },
    ]]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_real() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            Complex::<f32> {
                real: 1.0,
                imag: -2.0,
            },
            Complex::<f32> {
                real: -3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let result = tensor.real();
    let data = result.into_data();

    let expected = TensorData::from([[1.0, -3.0]]);
    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_imag() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            Complex::<f32> {
                real: 1.0,
                imag: -2.0,
            },
            Complex::<f32> {
                real: -3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let result = tensor.imag();
    let data = result.into_data();

    let expected = TensorData::from([[-2.0, 4.0]]);
    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_magnitude() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            Complex::<f32> {
                real: 3.0,
                imag: 4.0,
            }, // |3+4i| = 5
            Complex::<f32> {
                real: 0.0,
                imag: 1.0,
            }, // |0+1i| = 1
        ]]),
        &Default::default(),
    );

    let result = tensor.magnitude();
    let data = result.into_data();

    let expected = TensorData::from([[5.0, 1.0]]);
    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_phase() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            }, // arg(1+0i) = 0
            Complex::<f32> {
                real: 0.0,
                imag: 1.0,
            }, // arg(0+1i) = π/2
        ]]),
        &Default::default(),
    );

    let result = tensor.phase();
    let data = result.into_data();

    let expected = TensorData::from([[0.0, std::f32::consts::FRAC_PI_2]]);
    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_from_parts() {
    let result = TestTensor::<2>::from_parts(
        TensorData::from([[1.0_f32, 2.0], [3.0, 4.0]]),
        TensorData::from([[5.0_f32, 6.0], [7.0, 8.0]]),
    );
    let data = result.into_data();

    let expected = TensorData::from([
        [
            Complex::<f32> {
                real: 1.0,
                imag: 5.0,
            },
            Complex::<f32> {
                real: 2.0,
                imag: 6.0,
            },
        ],
        [
            Complex::<f32> {
                real: 3.0,
                imag: 7.0,
            },
            Complex::<f32> {
                real: 4.0,
                imag: 8.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_from_polar() {
    let magnitude =
        FloatTensor::<2>::from_data(TensorData::from([[1.0_f32, 2.0]]), &Default::default());

    let phase = FloatTensor::<2>::from_data(
        TensorData::from([[0.0_f32, std::f32::consts::FRAC_PI_2]]), // 0 and π/2 radians
        &Default::default(),
    );

    let result = TestTensor::<2>::from_polar(
        magnitude.into_primitive().tensor(),
        phase.into_primitive().tensor(),
    );
    let data = result.into_data();

    // r*cos(θ) + i*r*sin(θ)
    // 1*cos(0) + i*1*sin(0) = 1 + 0i
    // 2*cos(π/2) + i*2*sin(π/2) = 0 + 2i
    let expected = TensorData::from([[
        Complex::<f32> {
            real: 1.0,
            imag: 0.0,
        },
        Complex::<f32> {
            real: 0.0,
            imag: 2.0,
        },
    ]]);

    data.assert_approx_eq(&expected, Tolerance::<f32>::balanced());
}

#[test]
fn test_complex_exp() {
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            }, // exp(0) = 1
            Complex::<f32> {
                real: 0.0,
                imag: std::f32::consts::PI,
            }, // exp(iπ) = -1
        ]),
        &Default::default(),
    );

    let result = tensor.exp();

    // exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
    // exp(0 + 0i) = 1 * (1 + 0i) = 1 + 0i
    // exp(0 + πi) = 1 * (-1 + 0i) = -1 + 0i (approximately)
    let expected_data: Vec<f32> = interleaved_data_to_raw_float_data(result.into_data())
        .into_vec()
        .unwrap();
    assert!((expected_data[0] - 1.0).abs() < 1e-6); // real part of exp(0)
    assert!(expected_data[1].abs() < 1e-6); // imag part of exp(0)
    assert!((expected_data[2] + 1.0).abs() < 1e-5); // real part of exp(iπ), should be close to -1
    assert!(expected_data[3].abs() < 1e-5); // imag part of exp(iπ), should be close to 0
}

#[test]
fn test_complex_sin() {
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            }, // sin(0) = 0
            Complex::<f32> {
                real: std::f32::consts::FRAC_PI_2,
                imag: 0.0,
            }, // sin(π/2) = 1
        ]),
        &Default::default(),
    );

    let result = tensor.sin();

    let expected_data: Vec<f32> = interleaved_data_to_raw_float_data(result.into_data())
        .into_vec()
        .unwrap();
    assert!(expected_data[0].abs() < 1e-6); // real part of sin(0)
    assert!(expected_data[1].abs() < 1e-6); // imag part of sin(0)  
    assert!((expected_data[2] - 1.0).abs() < 1e-6); // real part of sin(π/2)
    assert!(expected_data[3].abs() < 1e-6); // imag part of sin(π/2)
}

#[test]
fn test_complex_cos() {
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            }, // cos(0) = 1
            Complex::<f32> {
                real: std::f32::consts::PI,
                imag: 0.0,
            }, // cos(π) = -1
        ]),
        &Default::default(),
    );

    let result = tensor.cos();

    let expected_data: Vec<f32> = interleaved_data_to_raw_float_data(result.into_data())
        .into_vec()
        .unwrap();
    assert!((expected_data[0] - 1.0).abs() < 1e-6); // real part of cos(0)
    assert!(expected_data[1].abs() < 1e-6); // imag part of cos(0)
    assert!((expected_data[2] + 1.0).abs() < 1e-5); // real part of cos(π)
    assert!(expected_data[3].abs() < 1e-5); // imag part of cos(π)
}

#[test]
fn test_complex_log() {
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            }, // log(1) = 0
            Complex::<f32> {
                real: std::f32::consts::E,
                imag: 0.0,
            }, // log(e) = 1
        ]),
        &Default::default(),
    );

    let result = tensor.log();

    let expected_data: Vec<f32> = interleaved_data_to_raw_float_data(result.into_data())
        .into_vec()
        .unwrap();
    assert!(expected_data[0].abs() < 1e-6); // real part of log(1)
    assert!(expected_data[1].abs() < 1e-6); // imag part of log(1)
    assert!((expected_data[2] - 1.0).abs() < 1e-5); // real part of log(e)
    assert!(expected_data[3].abs() < 1e-5); // imag part of log(e)
}

#[test]
fn test_complex_sqrt() {
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 4.0,
                imag: 0.0,
            }, // sqrt(4) = 2
            Complex::<f32> {
                real: -1.0,
                imag: 0.0,
            }, // sqrt(-1) = i
        ]),
        &Default::default(),
    );

    let result = tensor.sqrt();

    let expected_data: Vec<f32> = interleaved_data_to_raw_float_data(result.into_data())
        .into_vec()
        .unwrap();
    assert!((expected_data[0] - 2.0).abs() < 1e-6); // real part of sqrt(4)
    assert!(expected_data[1].abs() < 1e-6); // imag part of sqrt(4)
    assert!(expected_data[2].abs() < 1e-5); // real part of sqrt(-1)
    assert!((expected_data[3] - 1.0).abs() < 1e-5); // imag part of sqrt(-1)
}

#[test]
fn test_complex_matmul_identity() {
    // a = [[3+4i, 2+0i], [0-2i, 3+0i]]
    let a = TestTensor::<2>::from_data(
        TensorData::from([
            [
                Complex::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
                Complex::<f32> {
                    real: 2.0,
                    imag: 0.0,
                },
            ],
            [
                Complex::<f32> {
                    real: 0.0,
                    imag: -2.0,
                },
                Complex::<f32> {
                    real: 3.0,
                    imag: 0.0,
                },
            ],
        ]),
        &Default::default(),
    );

    // identity matrix
    let eye = TestTensor::<2>::from_data(
        TensorData::from([
            [
                Complex::<f32> {
                    real: 1.0,
                    imag: 0.0,
                },
                Complex::<f32> {
                    real: 0.0,
                    imag: 0.0,
                },
            ],
            [
                Complex::<f32> {
                    real: 0.0,
                    imag: 0.0,
                },
                Complex::<f32> {
                    real: 1.0,
                    imag: 0.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let expected = a.clone().into_data();
    let result = a.matmul(eye).into_data();

    result.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_acos() {
    // acos(1 + 0i) = 0, acos(0 + 0i) = π/2
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );

    let result = tensor.acos();

    let data: Vec<f32> = interleaved_data_to_raw_float_data(result.into_data())
        .into_vec()
        .unwrap();
    assert!(data[0].abs() < 1e-5, "re(acos(1)) = {}", data[0]);
    assert!(data[1].abs() < 1e-5, "im(acos(1)) = {}", data[1]);
    assert!(
        (data[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-5,
        "re(acos(0)) = {}",
        data[2]
    );
    assert!(data[3].abs() < 1e-5, "im(acos(0)) = {}", data[3]);
}

#[test]
fn test_complex_acosh() {
    // acosh(1 + 0i) = 0
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([Complex::<f32> {
            real: 1.0,
            imag: 0.0,
        }]),
        &Default::default(),
    );

    let result = tensor.acosh();

    let data: Vec<f32> = interleaved_data_to_raw_float_data(result.into_data())
        .into_vec()
        .unwrap();
    assert!(data[0].abs() < 1e-5, "re(acosh(1)) = {}", data[0]);
    assert!(data[1].abs() < 1e-5, "im(acosh(1)) = {}", data[1]);
}

#[test]
fn test_complex_asin() {
    // asin(0 + 0i) = 0, asin(1 + 0i) = π/2
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );

    let result = tensor.asin();

    let data: Vec<f32> = interleaved_data_to_raw_float_data(result.into_data())
        .into_vec()
        .unwrap();
    assert!(data[0].abs() < 1e-5, "re(asin(0)) = {}", data[0]);
    assert!(data[1].abs() < 1e-5, "im(asin(0)) = {}", data[1]);
    assert!(
        (data[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-5,
        "re(asin(1)) = {}",
        data[2]
    );
    assert!(data[3].abs() < 1e-5, "im(asin(1)) = {}", data[3]);
}

#[test]
fn test_complex_asinh() {
    // asinh(0 + 0i) = 0
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([Complex::<f32> {
            real: 0.0,
            imag: 0.0,
        }]),
        &Default::default(),
    );

    let result = tensor.asinh();

    let data: Vec<f32> = interleaved_data_to_raw_float_data(result.into_data())
        .into_vec()
        .unwrap();
    assert!(data[0].abs() < 1e-5, "re(asinh(0)) = {}", data[0]);
    assert!(data[1].abs() < 1e-5, "im(asinh(0)) = {}", data[1]);
}

#[test]
fn test_complex_atan() {
    // atan(0 + 0i) = 0, atan(1 + 0i) = π/4
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            Complex::<f32> {
                real: 0.0,
                imag: 0.0,
            },
            Complex::<f32> {
                real: 1.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );

    let result = tensor.atan();

    let data: Vec<f32> = interleaved_data_to_raw_float_data(result.into_data())
        .into_vec()
        .unwrap();
    assert!(data[0].abs() < 1e-5, "re(atan(0)) = {}", data[0]);
    assert!(data[1].abs() < 1e-5, "im(atan(0)) = {}", data[1]);
    assert!(
        (data[2] - std::f32::consts::FRAC_PI_4).abs() < 1e-5,
        "re(atan(1)) = {}",
        data[2]
    );
    assert!(data[3].abs() < 1e-5, "im(atan(1)) = {}", data[3]);
}

#[test]
fn test_complex_atanh() {
    // atanh(0 + 0i) = 0
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([Complex::<f32> {
            real: 0.0,
            imag: 0.0,
        }]),
        &Default::default(),
    );

    let result = tensor.atanh();

    let data: Vec<f32> = interleaved_data_to_raw_float_data(result.into_data())
        .into_vec()
        .unwrap();
    assert!(data[0].abs() < 1e-5, "re(atanh(0)) = {}", data[0]);
    assert!(data[1].abs() < 1e-5, "im(atanh(0)) = {}", data[1]);
}
