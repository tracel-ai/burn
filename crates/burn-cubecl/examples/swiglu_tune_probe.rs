//! Probe of the SwiGLU autotune on this machine: per key, the cold autotune (compile + race),
//! then every setting timed on device, beside the priority the autotune gives it.
use burn_cubecl::{
    CubeDevice, cubecl,
    kernel::swiglu::launch_tuned,
    tune::{Arithmetic, GroupPriority, Peaks},
};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::throughput::ThroughputKey;
use cubek::matmul::kernels::tensor::PlainTensor;
use cubek::std::Hardware;
use cubek::std::microkernel::KernelEstimate;
use cubek::std::routine::{EstimatedProblem, Routine};
use cubek::swiglu::kernels::{
    CubeSwigluRoutine, SwigluCall, SwigluKey, SwigluRoutineCandidate, SwigluRoutineProblem,
    SwigluTensors,
};
use std::time::Instant;

fn main() {
    let device = CubeDevice::rocm(0).expect("rocm device");
    let client = device.client();
    let hw = Hardware::from_client(&client);
    let iters: usize = std::env::var("PROBE_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let elem = match std::env::var("PROBE_ELEM").as_deref() {
        Ok("f32") => ElemType::Float(FloatKind::F32),
        Ok("bf16") => ElemType::Float(FloatKind::BF16),
        _ => ElemType::Float(FloatKind::F16),
    };
    let dims: Vec<usize> = std::env::var("PROBE_DIMS")
        .ok()
        .map(|v| v.split(',').map(|d| d.parse().unwrap()).collect())
        .unwrap_or(vec![2816, 8192, 14336]);
    let rows_list: Vec<usize> = std::env::var("PROBE_ROWS")
        .ok()
        .map(|v| v.split(',').map(|d| d.parse().unwrap()).collect())
        .unwrap_or(vec![1, 4, 16, 64, 256, 1024, 4096]);

    eprintln!(
        "hw: units_to_fill={:?} max_units_per_cube={} max_cube_count={:?}",
        hw.units_to_fill(),
        hw.properties.hardware.max_units_per_cube,
        hw.properties.hardware.max_cube_count
    );
    println!(
        "elem,dim,rows,setting,priority,dup_of,cube_dim,cube_count,us,rank,best_ratio,autotune_ms"
    );
    for &dim in &dims {
        for &rows in &rows_list {
            let both = PlainTensor::zeros(&client, &[rows, 2 * dim], [2 * dim, 1], elem);
            let call = SwigluCall {
                client: client.clone(),
                both,
            };
            let key = SwigluKey::from(&call.both);
            eprintln!("=== PROBE {key:?} ===");

            let start = Instant::now();
            let _ = launch_tuned(call.clone(), &device, GroupPriority::DEFAULT);
            cubecl::future::block_on(client.sync()).unwrap();
            let tune_ms = start.elapsed().as_secs_f64() * 1e3;

            let candidates = SwigluRoutineCandidate::list(CubeSwigluRoutine, &hw);
            let prepared: Vec<_> = candidates.iter().map(|c| c.prepared(&key)).collect();
            let rounds: usize = std::env::var("PROBE_ROUNDS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(7);
            let out = SwigluTensors::allocate_out(&client, &call.both);
            for candidate in &candidates {
                for _ in 0..3 {
                    candidate
                        .launch_contiguous(&call, out.clone())
                        .expect("launch");
                }
            }
            cubecl::future::block_on(client.sync()).unwrap();
            // Interleave the candidates over rounds and keep each one's fastest round, so a
            // clock change or another process lands on every candidate alike.
            let mut fastest = vec![f64::INFINITY; candidates.len()];
            for _ in 0..rounds {
                for (i, candidate) in candidates.iter().enumerate() {
                    let (_, profile) = client
                        .profile(
                            || {
                                for _ in 0..iters {
                                    candidate
                                        .launch_contiguous(&call, out.clone())
                                        .expect("launch");
                                }
                            },
                            "probe",
                        )
                        .unwrap();
                    let ticks = cubecl::future::block_on(profile.resolve()).expect("ticks");
                    let us = ticks.duration().as_secs_f64() * 1e6 / iters as f64;
                    fastest[i] = fastest[i].min(us);
                }
            }
            let mut rows_out = Vec::new();
            for (i, candidate) in candidates.iter().enumerate() {
                let (priority, dup_of, cube_dim, cube_count) = match &prepared[i] {
                    Some(p) => {
                        let dup = prepared[..i]
                            .iter()
                            .position(|e| e.as_ref() == Some(p))
                            .map(|j| j.to_string())
                            .unwrap_or_default();
                        // The autotune's own priority: the routine's estimate timed at the
                        // device's peaks, against the gate's roofline.
                        let problem = SwigluRoutineProblem::from(&key);
                        let (blueprint, grid) =
                            CubeSwigluRoutine::prepare(&hw, &problem, candidate.settings())
                                .expect("prepared above");
                        let mut estimate = KernelEstimate::from(&grid);
                        CubeSwigluRoutine::estimate(&problem, &blueprint, &grid, &mut estimate);
                        let f32 = ElemType::Float(FloatKind::F32);
                        let demand = problem.demand();
                        let peaks = Peaks::measured(
                            &client,
                            demand,
                            Arithmetic {
                                vector: ThroughputKey::compute(None, f32, f32),
                                matrix: None,
                            },
                        );
                        let prio = peaks.priority_of(&estimate, demand, &hw).get();
                        let pred = peaks
                            .time_of(&estimate, &hw)
                            .map(|t| t.as_secs_f64() * 1e6)
                            .unwrap_or(f64::NAN);
                        let floor = peaks
                            .floor_of(demand)
                            .map(|t| t.as_secs_f64() * 1e6)
                            .unwrap_or(f64::NAN);
                        (
                            format!("{prio}|{pred:.2}|{floor:.2}"),
                            dup,
                            format!("{:?}", p.launch().cube_dim).replace(',', ";"),
                            format!("{:?}", p.launch().cube_count).replace(',', ";"),
                        )
                    }
                    None => (
                        "0(declines)".into(),
                        String::new(),
                        String::new(),
                        String::new(),
                    ),
                };
                let us = fastest[i];
                rows_out.push((
                    candidate.to_string(),
                    priority,
                    dup_of,
                    cube_dim,
                    cube_count,
                    us,
                ));
            }
            let best = rows_out.iter().map(|r| r.5).fold(f64::INFINITY, f64::min);
            let mut order: Vec<f64> = rows_out.iter().map(|r| r.5).collect();
            order.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (name, priority, dup, cd, cc, us) in rows_out {
                let rank = order.iter().position(|v| *v == us).unwrap() + 1;
                println!(
                    "{elem:?},{dim},{rows},{name},{priority},{dup},{cd},{cc},{us:.2},{rank},{:.2},{tune_ms:.1}",
                    us / best
                );
            }
        }
    }
}
