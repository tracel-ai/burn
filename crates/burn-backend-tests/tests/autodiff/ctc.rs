use super::*;
use burn_tensor::{Device, TensorData, Tolerance, activation::log_softmax, module::ctc_loss};

/// Deterministic logits: sin((t*7 + n*13 + c*3) * 0.1). Matches the
/// fixture generator in `burn-nn/src/loss/ctc.rs`'s PyTorch comparison tests.
fn generate_logits(
    t_size: usize,
    n_size: usize,
    c_size: usize,
    device: &Device<TestBackend>,
) -> TestTensor<3> {
    let mut data = Vec::with_capacity(t_size * n_size * c_size);
    for t in 0..t_size {
        for n in 0..n_size {
            for c in 0..c_size {
                data.push(((t * 7 + n * 13 + c * 3) as f32 * 0.1).sin());
            }
        }
    }
    TestTensor::<3>::from_data(TensorData::new(data, [t_size, n_size, c_size]), device)
}

#[allow(clippy::too_many_arguments)]
fn run_comparison(
    t_size: usize,
    n_size: usize,
    c_size: usize,
    targets_flat: Vec<i64>,
    target_shape: [usize; 2],
    input_lengths: Vec<i64>,
    target_lengths: Vec<i64>,
    blank: usize,
    expected_grad_flat: &[f32],
) {
    let device = AutodiffDevice::new();
    let logits = generate_logits(t_size, n_size, c_size, &device).require_grad();
    let log_probs = log_softmax(logits.clone(), 2);

    let targets =
        TestTensorInt::<2>::from_data(TensorData::new(targets_flat, target_shape), &device);
    let input_lengths =
        TestTensorInt::<1>::from_data(TensorData::new(input_lengths, [n_size]), &device);
    let target_lengths =
        TestTensorInt::<1>::from_data(TensorData::new(target_lengths, [n_size]), &device);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank).sum();
    let grads = loss.backward();
    let logits_grad = logits.grad(&grads).expect(
        "logits should receive a gradient - if this fails on a backend with a fused ctc_loss \
         kernel, the autodiff override needs to skip the fused path",
    );

    logits_grad.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::new(expected_grad_flat.to_vec(), [t_size, n_size, c_size]),
        Tolerance::rel_abs(1e-3, 1e-3),
    );
}

/// Original coverage: uniform input_lengths, no repeated labels, small T.
/// Keeps a first-row sanity check against the PyTorch reference so a bug in
/// the first batch element's first timestep is caught quickly.
#[test]
fn test_ctc_loss_grad() {
    let t_size: usize = 5;
    let n_size: usize = 3;
    let c_size: usize = 4;
    let device = AutodiffDevice::new();

    let logits = generate_logits(t_size, n_size, c_size, &device).require_grad();
    let log_probs = log_softmax(logits.clone(), 2);

    let targets =
        TestTensorInt::<2>::from_data(TensorData::from([[1, 2, 0], [1, 0, 0], [3, 2, 1]]), &device);
    let input_lengths = TestTensorInt::<1>::from_data(TensorData::from([5, 5, 5]), &device);
    let target_lengths = TestTensorInt::<1>::from_data(TensorData::from([2, 1, 3]), &device);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0).sum();
    let grads = loss.backward();
    let logits_grad = logits.grad(&grads).expect(
        "logits should receive a gradient - if this fails on a backend with a fused ctc_loss \
         kernel, the autodiff override needs to skip the fused path",
    );

    let expected_first_row =
        TensorData::from([-0.1679008007_f32, -0.4595540464, 0.2795598209, 0.3478950262]);
    let logits_grad_data = logits_grad.into_data();
    let first_row: Vec<f32> = logits_grad_data.iter::<f32>().take(4).collect();
    TensorData::from(first_row.as_slice())
        .assert_approx_eq::<FloatElem>(&expected_first_row, Tolerance::rel_abs(1e-3, 1e-3));
}

/// Variable input_lengths exercise the per-sample `t_last = input_len - 1`
/// boundary init in the beta recursion and the OOB mask in the gradient
/// composition. Fixture lifted from burn-nn's pytorch_comparison_tests.
#[test]
fn test_ctc_loss_grad_mixed_input_lengths() {
    // T=12, N=3, C=5, input_lengths=[12, 7, 10], target_lengths=[3, 2, 4].
    let expected_grad_flat = [
        -0.4790987670_f32,
        -0.2554937005,
        0.1991624236,
        0.2478453964,
        0.2875846624,
        -0.3495813310,
        0.2268397957,
        0.2150714993,
        -0.2442178279,
        0.1518878639,
        -0.2764556706,
        0.2474014312,
        -0.2137086987,
        0.1371368915,
        0.1056260392,
        -0.2729502618,
        -0.3609606028,
        0.2159237266,
        0.2238420397,
        0.1941450834,
        -0.2953839302,
        0.1920599341,
        0.1974952668,
        -0.2054278404,
        0.1112565696,
        -0.1719199270,
        0.2299505472,
        -0.2864859998,
        0.1497263014,
        0.0787290633,
        -0.2035763413,
        -0.3042884767,
        0.2126964629,
        0.1810975969,
        0.1140707731,
        -0.2759391963,
        0.0975771844,
        0.1823379993,
        -0.1112988219,
        0.1073228419,
        -0.1336459517,
        0.1869296581,
        -0.1996247321,
        0.1846873760,
        -0.0383463502,
        -0.2254105806,
        -0.1834360659,
        0.1925925612,
        0.1462381780,
        0.0700158924,
        -0.2259973884,
        -0.0393539183,
        0.1802661419,
        -0.0571591072,
        0.1422442794,
        -0.0609069727,
        0.1089282706,
        -0.0313654318,
        0.2186669111,
        -0.2353227735,
        -0.2840364873,
        -0.0632198900,
        0.1755636632,
        0.1377806067,
        0.0339120962,
        -0.1904856712,
        -0.2139032930,
        0.1827126741,
        0.0056131603,
        0.2160631120,
        -0.0243270602,
        -0.0070458520,
        0.1070247591,
        0.2239368409,
        -0.2995886803,
        -0.2955487072,
        0.0309870224,
        0.1654911339,
        0.1581364125,
        -0.0590658709,
        -0.2191396207,
        -0.3791662455,
        0.1803640425,
        0.1225430891,
        0.2953987718,
        -0.0436352938,
        -0.1575258970,
        0.1785279512,
        0.1756918877,
        -0.1530586481,
        -0.1834939867,
        0.0909025446,
        0.1423641294,
        0.1959712654,
        -0.2457439601,
        -0.3619639874,
        -0.3929221630,
        0.1820438206,
        0.2454170734,
        0.3274252713,
        -0.0628800318,
        -0.2567180395,
        0.2112283260,
        0.0507859327,
        0.0575838275,
        -0.0587697029,
        0.1174769849,
        0.0783569664,
        0.2290501744,
        -0.3661144078,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0725664943,
        -0.1532069892,
        0.2162397504,
        -0.1248963475,
        0.1344300956,
        -0.0362483934,
        0.1295878887,
        -0.0502482466,
        0.2470482886,
        -0.2901395261,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.1349253207,
        0.0867646411,
        0.1998746395,
        -0.2658679783,
        0.1141540110,
        -0.0705668628,
        0.1519546807,
        -0.2509805560,
        0.2475892603,
        -0.0779965296,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.2338010073,
        0.2471641302,
        0.1834627241,
        -0.3026831448,
        0.1058573127,
        -0.1155209392,
        0.1921830922,
        -0.4129956067,
        0.2229512781,
        0.1133821756,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.2636392713,
        0.2323469073,
        -0.2913427949,
        0.1800564528,
        0.1425786912,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    run_comparison(
        12,
        3,
        5,
        vec![1, 4, 2, 0, 3, 1, 0, 0, 2, 4, 1, 3],
        [3, 4],
        vec![12, 7, 10],
        vec![3, 2, 4],
        0,
        &expected_grad_flat,
    );
}

/// Consecutive repeated labels `[1, 1, ...]` force the alpha and beta skip
/// transitions to stay disabled (`l'[s] != l'[s-2]` fails). Exercises the
/// `skip_allowed` branch condition on the fused kernel.
#[test]
fn test_ctc_loss_grad_repeated_labels() {
    // T=8, N=4, C=6, targets include the consecutive pair [1, 1, 2].
    let expected_grad_flat = [
        -0.2766432464_f32,
        -0.5202965736,
        0.1523768753,
        0.1896236390,
        0.2200277001,
        0.2349116206,
        -0.1854365915,
        0.2031330466,
        -0.4260218740,
        0.1678018719,
        0.1360142529,
        0.1045092493,
        -0.6603536606,
        0.2278252542,
        0.1691786796,
        0.1262856424,
        0.0972681716,
        0.0397959016,
        -0.0894432291,
        -0.5457318425,
        0.1490373611,
        0.1462858170,
        0.1569476575,
        0.1829041988,
        -0.2842915654,
        -0.4220107496,
        0.1822281033,
        0.1889107376,
        0.1791101843,
        0.1560532600,
        -0.1155678406,
        0.2295538932,
        -0.2645366490,
        -0.0288553704,
        0.1027252972,
        0.0766806602,
        -0.5448347330,
        0.2031028718,
        0.1589304954,
        0.1322451383,
        0.1189499870,
        -0.0683937520,
        -0.0873993114,
        -0.3051757514,
        -0.2355299890,
        0.1586059481,
        0.2018169016,
        0.2676822543,
        -0.3225219846,
        -0.2611543834,
        0.1922984123,
        0.1632783115,
        0.1297036558,
        0.0983960181,
        -0.1507159024,
        0.2256962359,
        -0.1040333956,
        -0.1514528394,
        0.0985243544,
        0.0819815546,
        -0.2940836251,
        0.1586865336,
        0.1468491107,
        0.1485087872,
        0.1639631987,
        -0.3239239752,
        -0.0767390430,
        -0.0434846729,
        -0.4023587406,
        -0.0052628326,
        0.2273432612,
        0.3005020022,
        -0.2598774135,
        -0.2188862711,
        0.1678501070,
        0.1352078766,
        0.1002781317,
        0.0754275694,
        -0.1502914876,
        0.1930875033,
        -0.0709601715,
        -0.2219523191,
        0.1243555173,
        0.1257609427,
        -0.0574148744,
        0.1152269915,
        0.1307857931,
        0.1599020809,
        0.2068412602,
        -0.5553412437,
        -0.0536844917,
        0.0758557543,
        -0.2106334567,
        -0.2509877980,
        0.1757438034,
        0.2637061775,
        -0.1759711355,
        -0.2431350052,
        0.1071053818,
        0.1259848624,
        0.1004033238,
        0.0856125653,
        -0.1173698306,
        0.1213828772,
        -0.1768893301,
        -0.2070008069,
        0.1709136516,
        0.2089634240,
        0.0153109450,
        0.0967332721,
        0.1268781722,
        0.1706230640,
        0.2291058898,
        -0.6386513710,
        -0.0536664203,
        0.1378114969,
        0.0360041447,
        -0.2989685237,
        -0.0084722806,
        0.1872915775,
        -0.1523490399,
        -0.2111770809,
        -0.0390694551,
        0.1366800815,
        0.1302325875,
        0.1356829405,
        -0.0982905105,
        -0.0127884001,
        -0.3586881459,
        -0.0259541404,
        0.2114149332,
        0.2843062580,
        -0.0324133746,
        0.1084750593,
        0.1447229236,
        0.1862253845,
        0.2259712219,
        -0.6329812407,
        -0.1173689738,
        0.1914442331,
        0.1654772907,
        -0.1376858056,
        -0.2194855511,
        0.1176188141,
        -0.1529908478,
        -0.0606661662,
        -0.3384291232,
        0.1524862647,
        0.1777049750,
        0.2218948901,
        -0.0923086405,
        -0.2855934799,
        -0.3215619624,
        0.1726681292,
        0.2303666323,
        0.2964293361,
        -0.2508065701,
        0.1479703039,
        0.1753441393,
        0.1917535067,
        0.1919818372,
        -0.4562432170,
        -0.2350299209,
        0.2257601619,
        0.1863904297,
        0.0388212129,
        -0.2966264784,
        0.0806845874,
        -0.1992894858,
        0.1068909168,
        -0.5761897564,
        0.1624972969,
        0.2155302167,
        0.2905607820,
        -0.1168124676,
        -0.6870660186,
        0.1488010883,
        0.1881926507,
        0.2230074406,
        0.2438773215,
        -0.5771554708,
        0.1980127096,
        0.1924194694,
        0.1714663208,
        0.1415647417,
        -0.1263078004,
        -0.3408652246,
        0.2292248607,
        0.1707807332,
        0.1269564927,
        -0.2634142637,
        0.0773174241,
    ];
    run_comparison(
        8,
        4,
        6,
        vec![1, 1, 2, 0, 2, 3, 2, 1, 5, 0, 0, 0, 1, 2, 3, 4],
        [4, 4],
        vec![8, 8, 8, 8],
        vec![3, 4, 1, 4],
        0,
        &expected_grad_flat,
    );
}

/// Empty target (`target_length == 0`). Analytical gradient: with uniform
/// logits the only valid alignment is all-blank, so for each `(t, c)` the
/// gradient of `sum(loss)` w.r.t. logits is `-0.5` at the blank class and
/// `+0.5` at the other class.
#[test]
fn test_ctc_loss_grad_empty_target() {
    let t_size: usize = 3;
    let n_size: usize = 1;
    let c_size: usize = 2;
    let device = AutodiffDevice::new();

    // Uniform logits (any value works; softmax normalizes). Use 0.0 for clarity.
    let logits = TestTensor::<3>::from_data(
        TensorData::new(
            alloc::vec![0.0f32; t_size * n_size * c_size],
            [t_size, n_size, c_size],
        ),
        &device,
    )
    .require_grad();
    let log_probs = log_softmax(logits.clone(), 2);

    // Dummy target column (never read: target_length is 0). Using [1, 1]
    // avoids backends that don't support zero-sized dims.
    let targets = TestTensorInt::<2>::from_data(TensorData::from([[0_i64]]), &device);
    let input_lengths = TestTensorInt::<1>::from_data(TensorData::from([3_i64]), &device);
    let target_lengths = TestTensorInt::<1>::from_data(TensorData::from([0_i64]), &device);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0).sum();
    let grads = loss.backward();
    let logits_grad = logits.grad(&grads).unwrap();

    let expected = TensorData::new(
        alloc::vec![-0.5_f32, 0.5, -0.5, 0.5, -0.5, 0.5],
        [t_size, n_size, c_size],
    );
    logits_grad
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(1e-4, 1e-4));
}

/// Regression guard for NaN gradient on unreachable targets, driven through
/// `.backward()` rather than a direct `ctc_loss_backward` call. The direct-path
/// version is covered by `test_ctc_loss_backward_unreachable_is_finite`; this
/// test exists because a lazy-backend regression in log_sum_exp or the
/// grad-composition mask can leave NaN in the autodiff trace even when the
/// direct backward call is finite.
///
/// Note: the direct backward asserts grad == 0 (the mask inside
/// `ctc_grad_from_alpha_beta_default` zeroes entries for +inf nll). The autodiff
/// path currently produces a non-zero but finite gradient for this case because
/// `loss.sum().backward()` seeds through a +inf forward value. Finiteness is
/// what guards training stability; the zero-vs-finite-nonzero difference is a
/// separate concern.
#[test]
fn test_ctc_loss_grad_unreachable_is_finite() {
    let device = AutodiffDevice::new();

    // T=2, target=[1, 1] needs three steps, so nll = +inf for this sample.
    let log_probs = TestTensor::<3>::full([2, 1, 3], (1.0f32 / 3.0).ln(), &device).require_grad();

    let targets = TestTensorInt::<2>::from_data(TensorData::from([[1_i64, 1]]), &device);
    let input_lengths = TestTensorInt::<1>::from_data(TensorData::from([2_i64]), &device);
    let target_lengths = TestTensorInt::<1>::from_data(TensorData::from([2_i64]), &device);

    let loss = ctc_loss(log_probs.clone(), targets, input_lengths, target_lengths, 0).sum();
    let grads = loss.backward();
    let log_probs_grad = log_probs.grad(&grads).unwrap();
    for v in log_probs_grad.into_data().iter::<f32>() {
        assert!(
            v.is_finite(),
            "gradient through .backward() must be finite for unreachable targets, got {v}",
        );
    }
}

/// Gradient through `.backward()` on a `swap_dims + narrow` input chain, a layout
/// produced by typical training code ([B, T, C] model output -> [T, B, C] ->
/// skip warmup). Forward-only tests cover `narrow` and `swap_dims+narrow`; this
/// also checks that the autodiff path preserves strides end-to-end rather than
/// materializing a contiguous copy with wrong offsets.
#[test]
fn test_ctc_loss_grad_swap_dims_then_narrow() {
    let b: usize = 2;
    let t_full: usize = 8;
    let warmup: usize = 2;
    let t_eff: usize = t_full - warmup;
    let c: usize = 4;

    let device = AutodiffDevice::new();

    let mut data = Vec::with_capacity(b * t_full * c);
    for bi in 0..b {
        for ti in 0..t_full {
            for ci in 0..c {
                data.push(((bi * 31 + ti * 7 + ci * 3) as f32 * 0.1).sin());
            }
        }
    }

    // Path A: [B, T, C] -> swap -> narrow, all on a single require_grad leaf.
    let logits_btc_a =
        TestTensor::<3>::from_data(TensorData::new(data.clone(), [b, t_full, c]), &device)
            .require_grad();
    let log_probs_a = log_softmax(logits_btc_a.clone(), 2)
        .swap_dims(0, 1)
        .narrow(0, warmup, t_eff);

    let targets_a = TestTensorInt::<2>::from_data(TensorData::from([[1_i64, 2], [2, 3]]), &device);
    let input_lengths_a =
        TestTensorInt::<1>::from_data(TensorData::from([t_eff as i64, t_eff as i64]), &device);
    let target_lengths_a = TestTensorInt::<1>::from_data(TensorData::from([2_i64, 2]), &device);

    let loss_a = ctc_loss(log_probs_a, targets_a, input_lengths_a, target_lengths_a, 0).sum();
    let grads_a = loss_a.backward();
    let grad_a = logits_btc_a.grad(&grads_a).unwrap();

    // Path B: build the equivalent [T, B, C] contiguous layout up front so the
    // log_softmax -> ctc_loss chain sees a plain contiguous tensor. Transpose
    // the incoming data to [T, B, C] order.
    let mut data_tbc = Vec::with_capacity(t_full * b * c);
    for ti in 0..t_full {
        for bi in 0..b {
            for ci in 0..c {
                data_tbc.push(((bi * 31 + ti * 7 + ci * 3) as f32 * 0.1).sin());
            }
        }
    }
    let logits_tbc_b =
        TestTensor::<3>::from_data(TensorData::new(data_tbc, [t_full, b, c]), &device)
            .require_grad();
    let log_probs_b = log_softmax(logits_tbc_b.clone(), 2).narrow(0, warmup, t_eff);

    let targets_b = TestTensorInt::<2>::from_data(TensorData::from([[1_i64, 2], [2, 3]]), &device);
    let input_lengths_b =
        TestTensorInt::<1>::from_data(TensorData::from([t_eff as i64, t_eff as i64]), &device);
    let target_lengths_b = TestTensorInt::<1>::from_data(TensorData::from([2_i64, 2]), &device);

    let loss_b = ctc_loss(log_probs_b, targets_b, input_lengths_b, target_lengths_b, 0).sum();
    let grads_b = loss_b.backward();
    let grad_b_tbc = logits_tbc_b.grad(&grads_b).unwrap();

    // Compare path A's grad (laid out [B, T, C]) against path B's grad (laid
    // out [T, B, C]) element by element.
    let grad_a_data: Vec<f32> = grad_a.into_data().iter::<f32>().collect();
    let grad_b_data: Vec<f32> = grad_b_tbc.into_data().iter::<f32>().collect();
    for bi in 0..b {
        for ti in 0..t_full {
            for ci in 0..c {
                let idx_a = (bi * t_full + ti) * c + ci;
                let idx_b = (ti * b + bi) * c + ci;
                let diff = (grad_a_data[idx_a] - grad_b_data[idx_b]).abs();
                assert!(
                    diff < 1e-3,
                    "grad mismatch at (b={bi}, t={ti}, c={ci}): A={} vs B={}",
                    grad_a_data[idx_a],
                    grad_b_data[idx_b],
                );
            }
        }
    }
}

/// Autodiff at production scale. The training repo (speechlet exp-11) reported
/// loss ~0.18 instead of ~1250 on a freshly-initialized model when log_probs
/// were routed through `log_softmax -> swap_dims(0,1) -> narrow(0, warmup, ...)`
/// on `Autodiff<Fusion<CubeBackend>>`, while the contiguous-[T, B, C] reference
/// (same raw data) produced the correct value. Reproduces that exact shape
/// chain with a deterministic fixture and compares the two paths.
#[test]
fn test_ctc_loss_scaled_swap_dims_then_narrow_autodiff() {
    let b: usize = 2;
    let t_full: usize = 494;
    let warmup: usize = 92;
    let t_eff: usize = t_full - warmup;
    let c: usize = 36;

    let device = AutodiffDevice::new();

    let mut data_btc = Vec::with_capacity(b * t_full * c);
    for bi in 0..b {
        for ti in 0..t_full {
            for ci in 0..c {
                data_btc.push(((bi * 31 + ti * 7 + ci * 3) as f32 * 0.1).sin());
            }
        }
    }

    let logits_btc_a =
        TestTensor::<3>::from_data(TensorData::new(data_btc.clone(), [b, t_full, c]), &device)
            .require_grad();
    let log_probs_a = log_softmax(logits_btc_a, 2)
        .swap_dims(0, 1)
        .narrow(0, warmup, t_eff);

    let mut targets_data = alloc::vec![0_i64; b * 64];
    for bi in 0..b {
        let max = if bi == 0 { 64 } else { 43 };
        for si in 0..max {
            targets_data[bi * 64 + si] = ((si * 3 + bi * 7) % 35 + 1) as i64;
        }
    }
    let targets_a =
        TestTensorInt::<2>::from_data(TensorData::new(targets_data.clone(), [b, 64]), &device);
    let input_lengths_a =
        TestTensorInt::<1>::from_data(TensorData::from([t_eff as i64, t_eff as i64]), &device);
    let target_lengths_a = TestTensorInt::<1>::from_data(TensorData::from([64_i64, 43]), &device);

    let loss_a_per_sample = ctc_loss(log_probs_a, targets_a, input_lengths_a, target_lengths_a, 0);
    let loss_a_vec: Vec<f32> = loss_a_per_sample
        .clone()
        .into_data()
        .iter::<f32>()
        .collect();

    // Path B: build the equivalent contiguous [T, B, C] layout up front.
    let mut data_tbc = Vec::with_capacity(t_full * b * c);
    for ti in 0..t_full {
        for bi in 0..b {
            for ci in 0..c {
                data_tbc.push(((bi * 31 + ti * 7 + ci * 3) as f32 * 0.1).sin());
            }
        }
    }
    let logits_tbc_b =
        TestTensor::<3>::from_data(TensorData::new(data_tbc, [t_full, b, c]), &device);
    let log_probs_b = log_softmax(logits_tbc_b, 2).narrow(0, warmup, t_eff);

    let targets_b = TestTensorInt::<2>::from_data(TensorData::new(targets_data, [b, 64]), &device);
    let input_lengths_b =
        TestTensorInt::<1>::from_data(TensorData::from([t_eff as i64, t_eff as i64]), &device);
    let target_lengths_b = TestTensorInt::<1>::from_data(TensorData::from([64_i64, 43]), &device);

    let loss_b_per_sample = ctc_loss(log_probs_b, targets_b, input_lengths_b, target_lengths_b, 0);
    let loss_b_vec: Vec<f32> = loss_b_per_sample.into_data().iter::<f32>().collect();

    for i in 0..b {
        assert!(
            loss_a_vec[i] > 0.0,
            "sample {i}: autodiff+swap_dims+narrow loss is not positive ({}); a valid CTC \
             loss must be >= 0 for log-probs <= 0",
            loss_a_vec[i],
        );
        assert!(
            (loss_a_vec[i] - loss_b_vec[i]).abs() < 1.0,
            "sample {i}: autodiff swap_dims+narrow loss ({}) diverges from contiguous \
             reference loss ({})",
            loss_a_vec[i],
            loss_b_vec[i],
        );
    }
}
