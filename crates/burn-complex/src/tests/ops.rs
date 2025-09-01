#[burn_tensor_testgen::testgen(complex)]
mod tests {
    use super::*;
    use burn_tensor::{
        Complex, Tensor, TensorData,
        backend::Backend,
        element::{Complex32, Complex64},
    };

    #[test]
    fn test_complex_zeros() {
        let tensor = Tensor::<TestBackend, 2, Complex>::zeros([2, 3], &Default::default());
        let data = tensor.into_data();

        let expected = TensorData::from([
            [
                Complex32 {
                    real: 0.0,
                    imag: 0.0,
                },
                Complex32 {
                    real: 0.0,
                    imag: 0.0,
                },
                Complex32 {
                    real: 0.0,
                    imag: 0.0,
                },
            ],
            [
                Complex32 {
                    real: 0.0,
                    imag: 0.0,
                },
                Complex32 {
                    real: 0.0,
                    imag: 0.0,
                },
                Complex32 {
                    real: 0.0,
                    imag: 0.0,
                },
            ],
        ]);

        data.assert_eq(&expected, false);
    }

    #[test]
    fn test_complex_ones() {
        let tensor = Tensor::<TestBackend, 2, Complex>::ones([2, 2], &Default::default());
        let data = tensor.into_data();

        let expected = TensorData::from([
            [
                Complex32 {
                    real: 1.0,
                    imag: 0.0,
                },
                Complex32 {
                    real: 1.0,
                    imag: 0.0,
                },
            ],
            [
                Complex32 {
                    real: 1.0,
                    imag: 0.0,
                },
                Complex32 {
                    real: 1.0,
                    imag: 0.0,
                },
            ],
        ]);

        data.assert_eq(&expected, false);
    }

    #[test]
    fn test_complex_from_data() {
        let data = TensorData::from([[
            Complex32 {
                real: 1.0,
                imag: 2.0,
            },
            Complex32 {
                real: 3.0,
                imag: 4.0,
            },
        ]]);

        let tensor =
            Tensor::<TestBackend, 2, Complex>::from_data(data.clone(), &Default::default());
        let result = tensor.into_data();

        result.assert_eq(&data, false);
    }

    #[test]
    fn test_complex_add() {
        let tensor1 = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 1.0,
                    imag: 2.0,
                },
                Complex32 {
                    real: 3.0,
                    imag: 4.0,
                },
            ]]),
            &Default::default(),
        );

        let tensor2 = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 5.0,
                    imag: 6.0,
                },
                Complex32 {
                    real: 7.0,
                    imag: 8.0,
                },
            ]]),
            &Default::default(),
        );

        let result = tensor1 + tensor2;
        let data = result.into_data();

        let expected = TensorData::from([[
            Complex32 {
                real: 6.0,
                imag: 8.0,
            }, // (1+5) + (2+6)i
            Complex32 {
                real: 10.0,
                imag: 12.0,
            }, // (3+7) + (4+8)i
        ]]);

        data.assert_eq(&expected, false);
    }

    #[test]
    fn test_complex_sub() {
        let tensor1 = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 5.0,
                    imag: 6.0,
                },
                Complex32 {
                    real: 7.0,
                    imag: 8.0,
                },
            ]]),
            &Default::default(),
        );

        let tensor2 = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 1.0,
                    imag: 2.0,
                },
                Complex32 {
                    real: 3.0,
                    imag: 4.0,
                },
            ]]),
            &Default::default(),
        );

        let result = tensor1 - tensor2;
        let data = result.into_data();

        let expected = TensorData::from([[
            Complex32 {
                real: 4.0,
                imag: 4.0,
            }, // (5-1) + (6-2)i
            Complex32 {
                real: 4.0,
                imag: 4.0,
            }, // (7-3) + (8-4)i
        ]]);

        data.assert_eq(&expected, false);
    }

    #[test]
    fn test_complex_mul() {
        let tensor1 = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 1.0,
                    imag: 2.0,
                },
                Complex32 {
                    real: 0.0,
                    imag: 1.0,
                },
            ]]),
            &Default::default(),
        );

        let tensor2 = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 3.0,
                    imag: 4.0,
                },
                Complex32 {
                    real: 0.0,
                    imag: 1.0,
                },
            ]]),
            &Default::default(),
        );

        let result = tensor1 * tensor2;
        let data = result.into_data();

        // (1+2i) * (3+4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        // (0+1i) * (0+1i) = (0*0 - 1*1) + (0*1 + 1*0)i = -1 + 0i
        let expected = TensorData::from([[
            Complex32 {
                real: -5.0,
                imag: 10.0,
            },
            Complex32 {
                real: -1.0,
                imag: 0.0,
            },
        ]]);

        data.assert_eq(&expected, false);
    }

    #[test]
    fn test_complex_div() {
        let tensor1 = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 1.0,
                    imag: 1.0,
                },
                Complex32 {
                    real: 2.0,
                    imag: 0.0,
                },
            ]]),
            &Default::default(),
        );

        let tensor2 = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 1.0,
                    imag: -1.0,
                },
                Complex32 {
                    real: 1.0,
                    imag: 0.0,
                },
            ]]),
            &Default::default(),
        );

        let result = tensor1 / tensor2;
        let data = result.into_data();

        // (1+1i) / (1-1i) = ((1+1i)(1+1i)) / ((1-1i)(1+1i)) = (1+2i-1) / (1+1) = 2i/2 = i
        // (2+0i) / (1+0i) = 2/1 = 2
        let expected = TensorData::from([[
            Complex32 {
                real: 0.0,
                imag: 1.0,
            },
            Complex32 {
                real: 2.0,
                imag: 0.0,
            },
        ]]);

        data.assert_eq(&expected, false);
    }

    #[test]
    fn test_complex_neg() {
        let tensor = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 1.0,
                    imag: -2.0,
                },
                Complex32 {
                    real: -3.0,
                    imag: 4.0,
                },
            ]]),
            &Default::default(),
        );

        let result = -tensor;
        let data = result.into_data();

        let expected = TensorData::from([[
            Complex32 {
                real: -1.0,
                imag: 2.0,
            },
            Complex32 {
                real: 3.0,
                imag: -4.0,
            },
        ]]);

        data.assert_eq(&expected, false);
    }

    #[test]
    fn test_complex_conj() {
        let tensor = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 1.0,
                    imag: -2.0,
                },
                Complex32 {
                    real: -3.0,
                    imag: 4.0,
                },
            ]]),
            &Default::default(),
        );

        let result = tensor.conj();
        let data = result.into_data();

        let expected = TensorData::from([[
            Complex32 {
                real: 1.0,
                imag: 2.0,
            }, // conjugate flips sign of imaginary part
            Complex32 {
                real: -3.0,
                imag: -4.0,
            },
        ]]);

        data.assert_eq(&expected, false);
    }

    #[test]
    fn test_complex_real() {
        let tensor = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 1.0,
                    imag: -2.0,
                },
                Complex32 {
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
        let tensor = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 1.0,
                    imag: -2.0,
                },
                Complex32 {
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
        let tensor = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 3.0,
                    imag: 4.0,
                }, // |3+4i| = 5
                Complex32 {
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
        let tensor = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 1.0,
                    imag: 0.0,
                }, // arg(1+0i) = 0
                Complex32 {
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
        let real = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &Default::default(),
        );

        let imag = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[5.0, 6.0], [7.0, 8.0]]),
            &Default::default(),
        );

        let result = Tensor::<TestBackend, 2, Complex>::from_parts(real, imag);
        let data = result.into_data();

        let expected = TensorData::from([
            [
                Complex32 {
                    real: 1.0,
                    imag: 5.0,
                },
                Complex32 {
                    real: 2.0,
                    imag: 6.0,
                },
            ],
            [
                Complex32 {
                    real: 3.0,
                    imag: 7.0,
                },
                Complex32 {
                    real: 4.0,
                    imag: 8.0,
                },
            ],
        ]);

        data.assert_eq(&expected, false);
    }

    #[test]
    fn test_complex_from_polar() {
        let magnitude = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0]]),
            &Default::default(),
        );

        let phase = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0, std::f32::consts::FRAC_PI_2]]), // 0 and π/2 radians
            &Default::default(),
        );

        let result = Tensor::<TestBackend, 2, Complex>::from_polar(magnitude, phase);
        let data = result.into_data();

        // r*cos(θ) + i*r*sin(θ)
        // 1*cos(0) + i*1*sin(0) = 1 + 0i
        // 2*cos(π/2) + i*2*sin(π/2) = 0 + 2i
        let expected = TensorData::from([[
            Complex32 {
                real: 1.0,
                imag: 0.0,
            },
            Complex32 {
                real: 0.0,
                imag: 2.0,
            },
        ]]);

        data.assert_eq(&expected, false);
    }

    #[test]
    fn test_complex_reshape() {
        let tensor = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 1.0,
                    imag: 2.0,
                },
                Complex32 {
                    real: 3.0,
                    imag: 4.0,
                },
                Complex32 {
                    real: 5.0,
                    imag: 6.0,
                },
                Complex32 {
                    real: 7.0,
                    imag: 8.0,
                },
            ]]),
            &Default::default(),
        );

        let result: Tensor<TestBackend, 3, Complex> = tensor.reshape([2, 2, 1]);
        let data = result.into_data();

        let expected = TensorData::from([
            [
                [Complex32 {
                    real: 1.0,
                    imag: 2.0,
                }],
                [Complex32 {
                    real: 3.0,
                    imag: 4.0,
                }],
            ],
            [
                [Complex32 {
                    real: 5.0,
                    imag: 6.0,
                }],
                [Complex32 {
                    real: 7.0,
                    imag: 8.0,
                }],
            ],
        ]);

        data.assert_eq(&expected, false);
    }

    #[test]
    fn test_complex_transpose() {
        let tensor = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([
                [
                    Complex32 {
                        real: 1.0,
                        imag: 2.0,
                    },
                    Complex32 {
                        real: 3.0,
                        imag: 4.0,
                    },
                ],
                [
                    Complex32 {
                        real: 5.0,
                        imag: 6.0,
                    },
                    Complex32 {
                        real: 7.0,
                        imag: 8.0,
                    },
                ],
            ]),
            &Default::default(),
        );

        let result = tensor.transpose();
        let data = result.into_data();

        let expected = TensorData::from([
            [
                Complex32 {
                    real: 1.0,
                    imag: 2.0,
                },
                Complex32 {
                    real: 5.0,
                    imag: 6.0,
                },
            ],
            [
                Complex32 {
                    real: 3.0,
                    imag: 4.0,
                },
                Complex32 {
                    real: 7.0,
                    imag: 8.0,
                },
            ],
        ]);

        data.assert_eq(&expected, false);
    }

    #[test]
    fn test_complex_exp() {
        let tensor = Tensor::<TestBackend, 1, Complex>::from_data(
            TensorData::from([
                Complex32 {
                    real: 0.0,
                    imag: 0.0,
                }, // exp(0) = 1
                Complex32 {
                    real: 0.0,
                    imag: std::f32::consts::PI,
                }, // exp(iπ) = -1
            ]),
            &Default::default(),
        );

        let result = tensor.exp();
        let data = result.into_data();

        // exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
        // exp(0 + 0i) = 1 * (1 + 0i) = 1 + 0i
        // exp(0 + πi) = 1 * (-1 + 0i) = -1 + 0i (approximately)
        let expected_data = data.convert::<f32>();
        assert!((expected_data.value[0] - 1.0).abs() < 1e-6); // real part of exp(0)
        assert!(expected_data.value[1].abs() < 1e-6); // imag part of exp(0)
        assert!((expected_data.value[2] + 1.0).abs() < 1e-5); // real part of exp(iπ), should be close to -1
        assert!(expected_data.value[3].abs() < 1e-5); // imag part of exp(iπ), should be close to 0
    }

    #[test]
    fn test_complex_sin() {
        let tensor = Tensor::<TestBackend, 1, Complex>::from_data(
            TensorData::from([
                Complex32 {
                    real: 0.0,
                    imag: 0.0,
                }, // sin(0) = 0
                Complex32 {
                    real: std::f32::consts::FRAC_PI_2,
                    imag: 0.0,
                }, // sin(π/2) = 1
            ]),
            &Default::default(),
        );

        let result = tensor.sin();
        let data = result.into_data();

        let expected_data = data.convert::<f32>();
        assert!(expected_data.value[0].abs() < 1e-6); // real part of sin(0)
        assert!(expected_data.value[1].abs() < 1e-6); // imag part of sin(0)  
        assert!((expected_data.value[2] - 1.0).abs() < 1e-6); // real part of sin(π/2)
        assert!(expected_data.value[3].abs() < 1e-6); // imag part of sin(π/2)
    }

    #[test]
    fn test_complex_cos() {
        let tensor = Tensor::<TestBackend, 1, Complex>::from_data(
            TensorData::from([
                Complex32 {
                    real: 0.0,
                    imag: 0.0,
                }, // cos(0) = 1
                Complex32 {
                    real: std::f32::consts::PI,
                    imag: 0.0,
                }, // cos(π) = -1
            ]),
            &Default::default(),
        );

        let result = tensor.cos();
        let data = result.into_data();

        let expected_data = data.convert::<f32>();
        assert!((expected_data.value[0] - 1.0).abs() < 1e-6); // real part of cos(0)
        assert!(expected_data.value[1].abs() < 1e-6); // imag part of cos(0)
        assert!((expected_data.value[2] + 1.0).abs() < 1e-5); // real part of cos(π)
        assert!(expected_data.value[3].abs() < 1e-5); // imag part of cos(π)
    }

    #[test]
    fn test_complex_log() {
        let tensor = Tensor::<TestBackend, 1, Complex>::from_data(
            TensorData::from([
                Complex32 {
                    real: 1.0,
                    imag: 0.0,
                }, // log(1) = 0
                Complex32 {
                    real: std::f32::consts::E,
                    imag: 0.0,
                }, // log(e) = 1
            ]),
            &Default::default(),
        );

        let result = tensor.log();
        let data = result.into_data();

        let expected_data = data.convert::<f32>();
        assert!(expected_data.value[0].abs() < 1e-6); // real part of log(1)
        assert!(expected_data.value[1].abs() < 1e-6); // imag part of log(1)
        assert!((expected_data.value[2] - 1.0).abs() < 1e-5); // real part of log(e)
        assert!(expected_data.value[3].abs() < 1e-5); // imag part of log(e)
    }

    #[test]
    fn test_complex_sqrt() {
        let tensor = Tensor::<TestBackend, 1, Complex>::from_data(
            TensorData::from([
                Complex32 {
                    real: 4.0,
                    imag: 0.0,
                }, // sqrt(4) = 2
                Complex32 {
                    real: -1.0,
                    imag: 0.0,
                }, // sqrt(-1) = i
            ]),
            &Default::default(),
        );

        let result = tensor.sqrt();
        let data = result.into_data();

        let expected_data = data.convert::<f32>();
        assert!((expected_data.value[0] - 2.0).abs() < 1e-6); // real part of sqrt(4)
        assert!(expected_data.value[1].abs() < 1e-6); // imag part of sqrt(4)
        assert!(expected_data.value[2].abs() < 1e-5); // real part of sqrt(-1)
        assert!((expected_data.value[3] - 1.0).abs() < 1e-5); // imag part of sqrt(-1)
    }

    #[test]
    fn test_complex_add_scalar() {
        let tensor = Tensor::<TestBackend, 2, Complex>::from_data(
            TensorData::from([[
                Complex32 {
                    real: 1.0,
                    imag: 2.0,
                },
                Complex32 {
                    real: 3.0,
                    imag: 4.0,
                },
            ]]),
            &Default::default(),
        );

        let scalar = Complex32 {
            real: 5.0,
            imag: 6.0,
        };
        let result = tensor + scalar;
        let data = result.into_data();

        let expected = TensorData::from([[
            Complex32 {
                real: 6.0,
                imag: 8.0,
            }, // (1+5) + (2+6)i
            Complex32 {
                real: 8.0,
                imag: 10.0,
            }, // (3+5) + (4+6)i
        ]]);

        data.assert_eq(&expected, false);
    }
}
