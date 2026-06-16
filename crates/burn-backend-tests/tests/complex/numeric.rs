use burn_tensor::Tensor;
use burn_tensor::Tolerance;
use burn_tensor::{ComplexScalar, TensorData};
use burn_tensor::{Distribution, Int};

#[test]
fn test_complex_add() {
    let tensor1 = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let tensor2 = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 6.0,
            },
            ComplexScalar::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ]]),
        &Default::default(),
    );

    let result = tensor1 + tensor2;
    let data = result.into_data();

    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 6.0,
            imag: 8.0,
        }, // (1+5) + (2+6)i
        ComplexScalar::<f32> {
            real: 10.0,
            imag: 12.0,
        }, // (3+7) + (4+8)i
    ]]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_sub() {
    let tensor1 = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 6.0,
            },
            ComplexScalar::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ]]),
        &Default::default(),
    );

    let tensor2 = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let result = tensor1 - tensor2;
    let data = result.into_data();

    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 4.0,
            imag: 4.0,
        }, // (5-1) + (6-2)i
        ComplexScalar::<f32> {
            real: 4.0,
            imag: 4.0,
        }, // (7-3) + (8-4)i
    ]]);

    data.assert_eq(&expected, false);
}

// #[test]
// fn test_complex_sub_float() {
//     let tensor1 = TestTensor::<2>::from_data(
//         TensorData::from([[
//             ComplexScalar::<f32> {
//                 real: 5.0,
//                 imag: 6.0,
//             },
//             ComplexScalar::<f32> {
//                 real: 7.0,
//                 imag: 8.0,
//             },
//         ]]),
//         &Default::default(),
//     );

//     let tensor2 = Tensor::<2, Float>::from_data(
//         TensorData::from([[
//             1.0, -3.0,
//         ]]),
//         &Default::default(),
//     );
      
    

//     let result = tensor1 - tensor2;
//     let data = result.into_data();

//     let expected = TensorData::from([[
//         ComplexScalar::<f32> {
//             real: 4.0,
//             imag: 4.0,
//         }, // (5-1) + (6-2)i
//         ComplexScalar::<f32> {
//             real: 4.0,
//             imag: 4.0,
//         }, // (7-3) + (8-4)i
//     ]]);

//     data.assert_eq(&expected, false);
// }



#[test]
fn test_complex_mul() {
    let tensor1 = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 1.0,
            },
        ]]),
        &Default::default(),
    );

    let tensor2 = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
            ComplexScalar::<f32> {
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
        ComplexScalar::<f32> {
            real: -5.0,
            imag: 10.0,
        },
        ComplexScalar::<f32> {
            real: -1.0,
            imag: 0.0,
        },
    ]]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_div() {
    let tensor1 = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 1.0,
            },
            ComplexScalar::<f32> {
                real: 2.0,
                imag: 0.0,
            },
        ]]),
        &Default::default(),
    );

    let tensor2 = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: -1.0,
            },
            ComplexScalar::<f32> {
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
        ComplexScalar::<f32> {
            real: 0.0,
            imag: 1.0,
        },
        ComplexScalar::<f32> {
            real: 2.0,
            imag: 0.0,
        },
    ]]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_neg() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: -2.0,
            },
            ComplexScalar::<f32> {
                real: -3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let result = -tensor;
    let data = result.into_data();

    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: -1.0,
            imag: 2.0,
        },
        ComplexScalar::<f32> {
            real: 3.0,
            imag: -4.0,
        },
    ]]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_mean_dim() {
    // a = [[3+4i, 2+0i], [0-2i, 3+0i]]
    let a = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
                ComplexScalar::<f32> {
                    real: 2.0,
                    imag: 0.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 0.0,
                    imag: -2.0,
                },
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 0.0,
                },
            ],
        ]),
        &Default::default(),
    );

    // mean along dim 0: col0 = (3+4i + 0-2i)/2 = 1.5+1i, col1 = (2+0i + 3+0i)/2 = 2.5+0i
    let result = a.mean_dim(0).into_data();

    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 1.5,
            imag: 1.0,
        },
        ComplexScalar::<f32> {
            real: 2.5,
            imag: 0.0,
        },
    ]]);

    result.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_add_scalar() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let scalar = ComplexScalar::<f64> {
        real: 5.0,
        imag: 6.0,
    };
    let result = tensor + scalar;
    let data = result.into_data();

    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 6.0,
            imag: 8.0,
        }, // (1+5) + (2+6)i
        ComplexScalar::<f32> {
            real: 8.0,
            imag: 10.0,
        }, // (3+5) + (4+6)i
    ]]);

    data.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_sub_scalar() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 6.0,
            },
            ComplexScalar::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ]]),
        &Default::default(),
    );

    let scalar = ComplexScalar::<f64> {
        real: 2.0,
        imag: 3.0,
    };
    let result = tensor - scalar;
    let data = result.into_data();

    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 3.0,
            imag: 3.0,
        }, // (5-2) + (6-3)i
        ComplexScalar::<f32> {
            real: 5.0,
            imag: 5.0,
        }, // (7-2) + (8-3)i
    ]]);

    data.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_mul_scalar() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let scalar = ComplexScalar::<f32> {
        real: 2.0,
        imag: 1.0,
    };
    let result = tensor * scalar;
    let data = result.into_data();

    // (1+2i) * (2+1i) = (1*2 - 2*1) + (1*1 + 2*2)i = 0 + 5i
    // (3+4i) * (2+1i) = (3*2 - 4*1) + (3*1 + 4*2)i = 2 + 11i
    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 2.0,
            imag: 5.0,
        },
        ComplexScalar::<f32> {
            real: 6.0,
            imag: 9.0,
        },
    ]]);

    data.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_div_scalar() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 4.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 6.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let scalar = ComplexScalar::<f64> {
        real: 2.0,
        imag: 0.0,
    };
    let result = tensor / scalar;
    let data = result.into_data();

    // (4+2i) / (2+0i) = 2+1i
    // (6+4i) / (2+0i) = 3+2i
    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 2.0,
            imag: 1.0,
        },
        ComplexScalar::<f32> {
            real: 3.0,
            imag: 2.0,
        },
    ]]);

    data.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_random() {
    let shape = [2usize, 3usize];
    let tensor =
        TestTensor::<2>::random(shape, Distribution::Uniform(0.0, 1.0), &Default::default());
    assert_eq!(tensor.shape().dims(), shape);
}

#[test]
fn test_complex_remainder() {
    let tensor1 = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 7.0,
            },
            ComplexScalar::<f32> {
                real: 8.0,
                imag: 9.0,
            },
        ]]),
        &Default::default(),
    );

    let tensor2 = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let result = tensor1 % tensor2;
    let data = result.into_data();

    // Component-wise: real=5%3=2, imag=7%4=3  and  real=8%3=2, imag=9%4=1
    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 2.0,
            imag: 3.0,
        },
        ComplexScalar::<f32> {
            real: 2.0,
            imag: 1.0,
        },
    ]]);

    data.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_remainder_scalar() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 7.0,
            },
            ComplexScalar::<f32> {
                real: 8.0,
                imag: 9.0,
            },
        ]]),
        &Default::default(),
    );

    let scalar = ComplexScalar::<f64> {
        real: 3.0,
        imag: 4.0,
    };
    let result = tensor % scalar;
    let data = result.into_data();

    // Component-wise per scalar: real%3, imag%4
    // (5+7i) % (3+4i): real=5%3=2, imag=7%4=3
    // (8+9i) % (3+4i): real=8%3=2, imag=9%4=1
    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 2.0,
            imag: 3.0,
        },
        ComplexScalar::<f32> {
            real: 2.0,
            imag: 1.0,
        },
    ]]);

    data.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_sum() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let result = tensor.sum().into_data();

    // (1+3) + (2+4)i = 4+6i
    let expected = TensorData::from([ComplexScalar::<f32> {
        real: 4.0,
        imag: 6.0,
    }]);

    result.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_sum_dim() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 2.0,
                },
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 5.0,
                    imag: 6.0,
                },
                ComplexScalar::<f32> {
                    real: 7.0,
                    imag: 8.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let result = tensor.sum_dim(0).into_data();

    // sum along dim 0: col0 = 1+2i+5+6i = 6+8i, col1 = 3+4i+7+8i = 10+12i
    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 6.0,
            imag: 8.0,
        },
        ComplexScalar::<f32> {
            real: 10.0,
            imag: 12.0,
        },
    ]]);

    result.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_prod() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 2.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 0.0,
            },
        ]]),
        &Default::default(),
    );

    let result = tensor.prod().into_data();

    // (2+0i) * (3+0i) = 6+0i
    let expected = TensorData::from([ComplexScalar::<f32> {
        real: 6.0,
        imag: 0.0,
    }]);

    result.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_prod_dim() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
                ComplexScalar::<f32> {
                    real: 2.0,
                    imag: 0.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 0.0,
                    imag: 1.0,
                },
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 0.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let result = tensor.prod_dim(0).into_data();

    // prod along dim 0:
    // col0: (1+1i)*(0+1i) = (0-1) + (1+0)i = -1+1i
    // col1: (2+0i)*(3+0i) = 6+0i
    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: -1.0,
            imag: 1.0,
        },
        ComplexScalar::<f32> {
            real: 6.0,
            imag: 0.0,
        },
    ]]);

    result.assert_approx_eq(&expected, Tolerance::<f32>::default());
}

#[test]
fn test_complex_mean() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let result = tensor.mean().into_data();

    // mean = (1+3)/2 + (2+4)/2 i = 2+3i
    let expected = TensorData::from([ComplexScalar::<f32> {
        real: 2.0,
        imag: 3.0,
    }]);

    result.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_powi_scalar() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 1.0,
            },
        ]]),
        &Default::default(),
    );

    let result = tensor.powi_scalar(2i32).into_data();

    // (1+0i)^2 = 1+0i
    // (0+1i)^2 = -1+0i
    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 1.0,
            imag: 0.0,
        },
        ComplexScalar::<f32> {
            real: -1.0,
            imag: 0.0,
        },
    ]]);

    result.assert_approx_eq(&expected, Tolerance::<f32>::default());
}

#[test]
fn test_complex_powi() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 2.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 1.0,
            },
        ]]),
        &Default::default(),
    );

    let exponents = Tensor::<2, Int>::from_data([[2i32, 2i32]], &Default::default());

    let result = tensor.powi(exponents).into_data();

    // (2+0i)^2 = 4+0i
    // (0+1i)^2 = -1+0i
    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 4.0,
            imag: 0.0,
        },
        ComplexScalar::<f32> {
            real: -1.0,
            imag: 0.0,
        },
    ]]);

    result.assert_approx_eq(&expected, Tolerance::<f32>::default());
}

#[test]
fn test_complex_matmul() {
    // A: [1, 1] shape (1x2)
    let a = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
        ]]),
        &Default::default(),
    );

    // B: [[1+2i], [3+4i]] shape (2x1)
    let b = TestTensor::<2>::from_data(
        TensorData::from([
            [ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            }],
            [ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            }],
        ]),
        &Default::default(),
    );

    let result = a.matmul(b).into_data();

    // [[1*(1+2i) + 1*(3+4i)]] = [[4+6i]]
    let expected = TensorData::from([[ComplexScalar::<f32> {
        real: 4.0,
        imag: 6.0,
    }]]);

    result.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_cumsum() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 2.0,
                },
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 5.0,
                    imag: 6.0,
                },
                ComplexScalar::<f32> {
                    real: 7.0,
                    imag: 8.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let result = tensor.cumsum(0).into_data();

    // cumsum along dim 0:
    // row 0: [1+2i, 3+4i]
    // row 1: [1+2i+5+6i, 3+4i+7+8i] = [6+8i, 10+12i]
    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 6.0,
                imag: 8.0,
            },
            ComplexScalar::<f32> {
                real: 10.0,
                imag: 12.0,
            },
        ],
    ]);

    result.assert_approx_eq(&expected, Tolerance::<f32>::strict());
}

#[test]
fn test_complex_cumprod() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 0.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 2.0,
                    imag: 0.0,
                },
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let result = tensor.cumprod(1).into_data();

    // cumprod along dim 1:
    // row 0: [1+1i, (1+1i)*(1+0i)] = [1+1i, 1+1i]
    // row 1: [2+0i, (2+0i)*(1+1i)] = [2+0i, 2+2i]
    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 1.0,
            },
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 1.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 2.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 2.0,
                imag: 2.0,
            },
        ],
    ]);

    result.assert_approx_eq(&expected, Tolerance::<f32>::default());
}

#[test]
fn test_complex_sign() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 2.0,
            },
        ]]),
        &Default::default(),
    );

    let result = tensor.sign().into_data();

    // sign(3+4i) = (3+4i)/|(3+4i)| = (3+4i)/5 = 0.6+0.8i
    // sign(0+2i) = (0+2i)/2 = 0+1i
    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 0.6,
            imag: 0.8,
        },
        ComplexScalar::<f32> {
            real: 0.0,
            imag: 1.0,
        },
    ]]);

    result.assert_approx_eq(&expected, Tolerance::<f32>::default());
}
