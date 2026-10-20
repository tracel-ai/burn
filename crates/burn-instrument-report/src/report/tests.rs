use std::time::Duration;

use burn_instrument::{
    Bounds, Breakdown, CollectedPass, Component, ComponentPath, ComputePeak, Kernel, Peaks,
    Unplaced, Variant,
};
use burn_tensor::FloatDType;

use super::table::*;
use super::{HostSplit, Report, TimeReading};
use crate::roofline::{Bound, Ladder};

fn peaks() -> Peaks {
    Peaks {
        copy_bytes_per_s: Some(198.9e9),
        flops: Some(ComputePeak {
            per_s: 49.3e12,
            dtype: FloatDType::F16,
        }),
        ..Peaks::default()
    }
}

/// A pass with a small tree, **built rather than measured**.
///
/// The table is a function of a [`CollectedPass`], so a rendering test
/// needs one, not a run that produces one: collecting a pass here would
/// leave every number at whatever this machine happened to do. What the
/// collector produces from a real pass is `burn-instrument`'s to test;
/// these tests own the rendering, and their input is stated.
fn pass() -> CollectedPass {
    CollectedPass {
        device: "cpu".to_string(),
        peaks: Some(peaks()),
        clean_wall: Some(Duration::from_micros(2_000)),
        breakdown: Breakdown {
            tree: tree(),
            untagged: vec![kernel("Reshape", 1, 20)],
            end_of_pass: Vec::new(),
            outside_wait: Duration::ZERO,
            host_timed: false,
            // Sized against a 2.0 ms wall that its components are apportioned
            // onto; the in-place reading has tests of its own.
            timed_alone: true,
            unplaced: Unplaced::default(),
            unfused: false,
        },
    }
}

/// The components of [`pass`]: a two-layer stack between an embedding and a
/// head, priced the way the components price themselves — a composite below
/// the sum of its parts, and the leaves carrying the kernels.
///
/// The floors are sized against [`peaks`]: bytes ÷ 198.9 GB/s and flops ÷
/// 49.3 TFLOP/s, so a leaf here is a tenth to a quarter of a millisecond
/// and the rungs land inside the 2.0 ms the pass took. Numbers a thousand
/// times smaller would render as `0.0` and make the ladder read as
/// incoherent, which is a fact about the fixture and not about the code.
/// `attention engine` is the one component held against `flops`, so a table has
/// one of each wall.
fn tree() -> Vec<Component> {
    vec![
        Component {
            label: "embedding".into(),
            bounds: Some(Bounds {
                read: 15_000_000,
                write: 5_000_000,
                flops: 0,
            }),
            kernels: vec![kernel("Gather", 1, 50)],
            ..component()
        },
        Component {
            label: "transformer block".into(),
            observed: None,
            entries: 2,
            bounds: Some(Bounds {
                read: 82_500_000,
                write: 27_500_000,
                flops: 3_000_000_000,
            }),
            children: vec![
                Component {
                    label: "attn norm".into(),
                    observed: None,
                    entries: 2,
                    bounds: Some(Bounds {
                        read: 15_000_000,
                        write: 5_000_000,
                        flops: 0,
                    }),
                    kernels: vec![kernel("RmsNormKernel", 2, 70)],
                    ..component()
                },
                Component {
                    label: "attention".into(),
                    observed: None,
                    entries: 2,
                    bounds: Some(Bounds {
                        read: 45_000_000,
                        write: 15_000_000,
                        flops: 900_000_000,
                    }),
                    children: vec![
                        Component {
                            label: "query".into(),
                            observed: None,
                            entries: 2,
                            bounds: Some(Bounds {
                                read: 30_000_000,
                                write: 10_000_000,
                                flops: 300_000_000,
                            }),
                            kernels: vec![kernel("Matmul", 2, 160)],
                            ..component()
                        },
                        Component {
                            label: "attention engine".into(),
                            observed: None,
                            entries: 2,
                            bounds: Some(Bounds {
                                read: 30_000_000,
                                write: 10_000_000,
                                flops: 12_000_000_000,
                            }),
                            kernels: vec![kernel("FlashAttention", 2, 220)],
                            ..component()
                        },
                    ],
                    ..component()
                },
                Component {
                    label: "feed forward".into(),
                    observed: None,
                    entries: 2,
                    bounds: Some(Bounds {
                        read: 45_000_000,
                        write: 15_000_000,
                        flops: 2_000_000_000,
                    }),
                    children: vec![
                        Component {
                            label: "gate".into(),
                            observed: None,
                            entries: 2,
                            bounds: Some(Bounds {
                                read: 30_000_000,
                                write: 10_000_000,
                                flops: 800_000_000,
                            }),
                            kernels: vec![kernel("Matmul", 2, 240)],
                            ..component()
                        },
                        Component {
                            label: "down".into(),
                            observed: None,
                            entries: 2,
                            bounds: Some(Bounds {
                                read: 30_000_000,
                                write: 10_000_000,
                                flops: 800_000_000,
                            }),
                            kernels: vec![kernel("Matmul", 2, 240)],
                            ..component()
                        },
                    ],
                    ..component()
                },
            ],
            ..component()
        },
        Component {
            label: "final norm".into(),
            bounds: Some(Bounds {
                read: 7_500_000,
                write: 2_500_000,
                flops: 0,
            }),
            kernels: vec![kernel("RmsNormKernel", 1, 30)],
            ..component()
        },
        Component {
            label: "lm head".into(),
            bounds: Some(Bounds {
                read: 22_500_000,
                write: 7_500_000,
                flops: 1_000_000_000,
            }),
            children: vec![Component {
                label: "head projection".into(),
                bounds: Some(Bounds {
                    read: 30_000_000,
                    write: 10_000_000,
                    flops: 1_000_000_000,
                }),
                kernels: vec![kernel("Matmul", 1, 260)],
                ..component()
            }],
            ..component()
        },
    ]
}

/// A component is held to its declared floor plus the traffic observed beyond
/// it, and its heading says so.
#[test]
fn observed_traffic_raises_the_floor_and_marks_the_row() {
    let declared = Bounds {
        read: 100,
        write: 10,
        flops: 50,
    };
    let observed = Bounds {
        read: 300,
        write: 0,
        flops: 0,
    };
    let plain = Component {
        label: "routed experts".into(),
        bounds: Some(declared),
        ..component()
    };
    let read = Component {
        observed: Some(observed),
        ..plain.clone()
    };

    assert_eq!(Report::held(&plain), Some(declared));
    assert_eq!(Report::held(&read), Some(declared + observed));
    assert_eq!(heading(&plain), "routed experts");
    assert_eq!(heading(&read), "routed experts · floor read from the pass");
}

/// An empty component, so a fixture states only the fields it is about.
fn component() -> Component {
    Component {
        label: String::new(),
        variants: Vec::new(),
        sites: Vec::new(),
        bounds: None,
        observed: None,
        entries: 1,
        children: Vec::new(),
        kernels: Vec::new(),
        own_wait: Duration::ZERO,
    }
}

fn kernel(name: &str, launches: usize, micros: u64) -> Kernel {
    Kernel {
        name: name.to_string(),
        launches,
        total: Duration::from_micros(micros),
        fuses: Vec::new(),
        fuses_own: false,
    }
}

/// Without an uncollected run there is no host to split off, and a pass
/// timed alone has nothing to apportion its components onto: they stay the
/// serialized kernels' own times, and the table says only their proportions
/// mean anything rather than print a `framework` line it cannot know.
#[test]
fn a_pass_without_a_clean_run_splits_nothing_and_says_so() {
    let pass = CollectedPass {
        clean_wall: None,
        ..pass()
    };
    let report = Report::new(&pass);
    assert_eq!(report.framework(), None);
    assert_eq!(
        report.wall(),
        pass.breakdown.total_time(),
        "the ladder starts from what the device measured"
    );
    let embedding = &pass.breakdown.tree[0];
    assert_eq!(
        report.apportioned(embedding),
        embedding.total_time(),
        "nothing to apportion onto"
    );

    let table = report.to_string();
    assert!(table.contains("no uncollected run measured"), "{table}");
    assert!(
        table.contains("only the proportions mean anything"),
        "{table}"
    );
    assert!(
        table.contains("not separable: no uncollected run was measured"),
        "{table}"
    );
    assert!(
        !table.contains("\n  wall "),
        "no clean wall to print:\n{table}"
    );
}

/// A pass nobody probed is timed and not scored, and the header says why
/// every roofline column is blank.
#[test]
fn an_unprobed_pass_says_why_nothing_is_scored() {
    let pass = CollectedPass {
        peaks: None,
        ..pass()
    };
    let report = Report::new(&pass);
    for (_, component) in pass.breakdown.tree.iter().flat_map(|root| root.walk()) {
        assert!(report.score_of(component).is_none(), "{}", component.label);
    }
    let table = report.to_string();
    assert!(table.contains("not probed"), "{table}");
    // A dtype is a fact about a compute peak, and there is none to state.
    let header = table.lines().next().expect("the device line");
    assert_eq!(header, "device   cpu · unprobed · unprobed", "{table}");
}

/// Each reason a pass has no host/device split is its own case, and only a pipelined pass
/// whose kernels fit inside the uncollected run splits: the table explains the case it is
/// handed rather than re-deriving it.
#[test]
fn a_host_split_names_why_it_has_none() {
    let in_place = |wall: Option<u64>, device: u64| {
        let mut pass = pass();
        pass.clean_wall = wall.map(Duration::from_micros);
        pass.breakdown.timed_alone = false;
        pass.breakdown.untagged.clear();
        pass.breakdown.tree = vec![Component {
            label: "only".into(),
            kernels: match device {
                0 => Vec::new(),
                micros => vec![kernel("Only", 1, micros)],
            },
            ..component()
        }];
        HostSplit::new(&pass)
    };
    assert_eq!(
        in_place(Some(2_000), 1_500),
        HostSplit::Measured(Duration::from_micros(500))
    );
    assert_eq!(in_place(None, 1_500), HostSplit::NoCleanRun);
    assert_eq!(in_place(Some(2_000), 0), HostSplit::NothingTimed);
    assert_eq!(in_place(Some(1_000), 1_500), HostSplit::ClocksDisagree);
    assert_eq!(HostSplit::new(&pass()), HostSplit::TimedAlone);
}

/// A component's milliseconds are apportioned only where the kernels ran alone and there is
/// an uncollected run to apportion onto.
#[test]
fn a_time_reading_apportions_only_onto_a_clean_run() {
    let reading = |timed_alone: bool, wall: Option<u64>| {
        let mut pass = pass();
        pass.breakdown.timed_alone = timed_alone;
        pass.clean_wall = wall.map(Duration::from_micros);
        TimeReading::new(&pass)
    };
    assert_eq!(reading(false, Some(2_000)), TimeReading::InPlace);
    assert_eq!(reading(false, None), TimeReading::InPlace);
    assert_eq!(
        reading(true, Some(2_000)),
        TimeReading::ApportionedOnto(Duration::from_micros(2_000))
    );
    assert_eq!(reading(true, None), TimeReading::Serialized);
}

/// Print a real table, so its shape can be read rather than only asserted
/// on. A column that lines up in a test never proves it lines up on a page.
#[test]
#[ignore = "a reporting aid, not an assertion"]
fn show() {
    println!("{}", Report::new(&pass()));
}

#[test]
fn a_table_carries_every_column_a_reader_acts_on() {
    let pass = pass();
    let table = Report::new(&pass).to_string();

    for column in [
        "component",
        "ms",
        "share",
        "bound",
        "achieved",
        "floor",
        "gain",
    ] {
        assert!(table.contains(column), "missing {column}:\n{table}");
    }
    assert!(
        table.contains("pass"),
        "what the components break down:\n{table}"
    );
    assert!(table.contains("framework"), "the host half:\n{table}");
    assert!(
        table.contains("how much is on the table"),
        "the ladder:\n{table}"
    );
}

#[test]
fn an_unfused_breakdown_says_so_on_the_table() {
    let mut pass = pass();
    pass.breakdown.unfused = true;
    let table = Report::new(&pass).to_string();
    assert!(
        table.contains("graph     unfused"),
        "a reader must not take an unfused breakdown for a fused one:\n{table}"
    );
}

/// A fused table says so, and a component whose work all ran in a kernel
/// its parent carries reads `fused ↓` with no floor — not `0.0`, which
/// would read as a component that cost nothing, and not its declared
/// floor, which the kernel below it now counts.
#[test]
fn a_component_whose_work_fused_into_its_parent_says_so() {
    let mut pass = pass();
    // Every op the norm registered fused with the query projection's, so
    // the kernel that did both is on the block and none is on the norm.
    pass.breakdown.tree[1].children[0].kernels.clear();
    pass.breakdown.tree[1].kernels = vec![Kernel {
        fuses: vec![
            ComponentPath::new(["attn norm"]),
            ComponentPath::new(["attention", "query"]),
        ],
        ..kernel("ElemwiseMatmul", 2, 70)
    }];

    let table = Report::new(&pass).to_string();
    assert!(table.contains("graph     fused"), "{table}");
    let norm = table
        .lines()
        .find(|line| line.contains("attn norm"))
        .expect("the norm's component");
    assert!(norm.contains("fused ↓"), "{norm}");
    assert!(!norm.contains("bytes"), "and sets its floor aside:\n{norm}");

    // A component with no kernel that no parent's kernel names did no
    // device work — a view, a reshape — and its zero is its time. A
    // `fused ↓` there would point at a kernel that carries nothing of it.
    pass.breakdown.tree[1].kernels.clear();
    let table = Report::new(&pass).to_string();
    let norm = table
        .lines()
        .find(|line| line.contains("attn norm"))
        .expect("the norm's component");
    assert!(!norm.contains("fused ↓"), "{norm}");
}

/// A fused kernel counts its work once. Where every component it names
/// ran nothing else, their floors are set aside and the kernel stands at
/// what it took on the parent that holds it; where every one still ran
/// kernels of its own, they stay priced and the kernel is not counted
/// again on top of their floors.
#[test]
fn a_fused_kernel_on_a_parent_is_not_counted_twice() {
    let pass = pass();
    let mut fused = pass.clone();
    let norm = std::mem::take(&mut fused.breakdown.tree[1].children[0].kernels);
    let query = std::mem::take(&mut fused.breakdown.tree[1].children[1].children[0].kernels);
    let total = norm.iter().chain(&query).map(|kernel| kernel.total).sum();
    let carried = Kernel {
        fuses: vec![
            ComponentPath::new(["attn norm"]),
            ComponentPath::new(["attention", "query"]),
        ],
        total,
        ..kernel("ElemwiseMatmul", 2, 0)
    };
    fused.breakdown.tree[1].kernels = vec![carried.clone()];

    let rendered = Report::new(&fused);
    let block = &fused.breakdown.tree[1];
    assert_eq!(
        rendered.unpriced_kernel_time(block),
        total,
        "their work is the kernel's, at what it took"
    );
    assert_eq!(
        rendered.deepest_floor(&block.children[0]),
        Duration::ZERO,
        "and the norm's floor is not counted beside it"
    );

    // Both still ran a kernel of their own besides: still priced, so the
    // shared kernel is counted at their floors and not again.
    let mut partly = pass.clone();
    partly.breakdown.tree[1].kernels = vec![carried];
    let rendered = Report::new(&partly);
    assert_eq!(
        rendered.unpriced_kernel_time(&partly.breakdown.tree[1]),
        Duration::ZERO
    );
}

/// A residual add fused with its child's work lands on the parent naming
/// only the child: the child ran nothing, so its floor is set aside and
/// the kernel — the child's work and the parent's own — counts once, at
/// what it took.
#[test]
fn a_parent_fused_with_its_child_counts_shared_work_once() {
    let pass = pass();
    let mut fused = pass.clone();
    let norm = std::mem::take(&mut fused.breakdown.tree[1].children[0].kernels);
    let total: Duration =
        norm.iter().map(|kernel| kernel.total).sum::<Duration>() + Duration::from_micros(5);
    fused.breakdown.tree[1].kernels = vec![Kernel {
        fuses: vec![ComponentPath::new(["attn norm"])],
        fuses_own: true,
        total,
        ..kernel("ElemwiseFuse", 1, 0)
    }];

    let rendered = Report::new(&fused);
    let block = &fused.breakdown.tree[1];
    assert!(rendered.is_fused_elsewhere(&block.children[0]));
    assert_eq!(rendered.unpriced_kernel_time(block), total);
    assert_eq!(rendered.deepest_floor(&block.children[0]), Duration::ZERO);
    let table = rendered.to_string();
    assert!(
        table.contains("ElemwiseFuse ×1 · fuses ↑ attn norm"),
        "{table}"
    );
}

/// A block with work from outside every component is recorded untagged,
/// naming the component whose work it carried by its path from the root: that
/// component reads `fused ↓` with its floor set aside, as it would under a
/// parent's kernel.
#[test]
fn a_row_whose_work_ran_in_an_untagged_kernel_says_so() {
    let mut pass = pass();
    let norm = std::mem::take(&mut pass.breakdown.tree[1].children[0].kernels);
    pass.breakdown.untagged.push(Kernel {
        fuses: vec![ComponentPath::new(["transformer block", "attn norm"])],
        ..norm[0].clone()
    });

    let rendered = Report::new(&pass);
    let block = &pass.breakdown.tree[1];
    assert!(rendered.is_fused_elsewhere(&block.children[0]));
    assert_eq!(rendered.deepest_floor(&block.children[0]), Duration::ZERO);
    let table = rendered.to_string();
    let component = table
        .lines()
        .find(|line| line.contains("attn norm") && !line.contains("fuses"))
        .expect("the norm's component");
    assert!(component.contains("fused ↓"), "{component}");
    assert!(
        !component.contains("bytes"),
        "and sets its floor aside:\n{component}"
    );
}

/// A kernel that carried a parent's own work stands at what it took even
/// where the child it names is still priced — it ran a kernel of its own
/// besides — since no floor below prices the parent's part of it.
#[test]
fn a_parents_own_work_fused_with_a_priced_child_still_counts() {
    let mut pass = pass();
    let residual = Kernel {
        fuses: vec![ComponentPath::new(["attn norm"])],
        fuses_own: true,
        ..kernel("ElemwiseFuse", 1, 15)
    };
    pass.breakdown.tree[1].kernels = vec![residual.clone()];

    let rendered = Report::new(&pass);
    let block = &pass.breakdown.tree[1];
    assert!(!rendered.is_fused_elsewhere(&block.children[0]));
    assert_eq!(rendered.unpriced_kernel_time(block), residual.total);
}

/// A kernel that carried out several components' work says whose, on the
/// component that holds all of it.
#[test]
fn a_fused_kernel_names_the_components_it_carried_out() {
    let mut pass = pass();
    pass.breakdown.tree[1].kernels = vec![Kernel {
        fuses: vec![
            ComponentPath::new(["attn norm"]),
            ComponentPath::new(["attention", "query"]),
        ],
        ..kernel("ElemwiseMatmul", 2, 90)
    }];

    let table = Report::new(&pass).to_string();
    assert!(
        table.contains("ElemwiseMatmul ×2 · fuses ↑ attn norm + attention › query"),
        "{table}"
    );
}

/// The ladder's "kernels at their own roofs" is a sum of floors, and a
/// floor is the declaration's. Taking it from the component's score dropped every
/// component whose work fused forward, since a score needs measured time — and
/// pulled the rung below what the pass's own declarations cost.
#[test]
fn a_floor_counts_on_the_ladder_whoever_ran_its_work() {
    let pass = pass();
    let before = Report::new(&pass).ladder();

    let mut fused = pass.clone();
    let moved = std::mem::take(&mut fused.breakdown.tree[1].children[0].kernels);
    // Onto the query projection, the leaf that sealed them.
    fused.breakdown.tree[1].children[1].children[0]
        .kernels
        .extend(moved);
    let after = Report::new(&fused).ladder();

    assert_eq!(
        before.kernels, after.kernels,
        "the same declarations are the same floors"
    );
    assert_eq!(before.fused, after.fused);
}

/// Measured in place, a component's milliseconds are its kernels' own and the
/// clean pass splits into the device's time and the host's: the fixture's
/// kernels come to 1.29 ms of a 2.0 ms pass, so the device waited 0.71 ms.
#[test]
fn a_pass_timed_in_place_splits_the_clean_pass_into_device_and_host() {
    let mut pass = pass();
    pass.breakdown.timed_alone = false;
    assert_eq!(
        Report::new(&pass).framework(),
        Some(Duration::from_micros(710))
    );

    let rendered = Report::new(&pass);
    let head = &pass.breakdown.tree[3].children[0];
    assert_eq!(
        rendered.apportioned(head),
        Duration::from_micros(260),
        "the kernel's own time, not a share of the wall"
    );
    assert_eq!(rendered.ladder().host, Duration::from_micros(710));

    let table = rendered.to_string();
    let framework = table
        .lines()
        .find(|line| line.trim_start().starts_with("framework"))
        .expect("the framework line");
    assert!(framework.contains("0.7 ms"), "{framework}");
    assert!(framework.contains("35.5%"), "{framework}");
    assert!(table.contains("on device"), "{table}");
    assert!(table.contains("timed in place"), "{table}");
}

/// The clean pass's host time is split by how long the device waited
/// before each component's kernels, and each component's own share is on
/// the worklist under the lever that recovers it.
#[test]
fn the_host_time_is_split_by_where_the_device_waited() {
    let mut pass = pass();
    pass.breakdown.timed_alone = false;
    let micros = Duration::from_micros;
    // The query's launches kept the device waiting three times as long as
    // the head's: 30 µs against 10 µs of the collected pass's idle time.
    pass.breakdown.tree[1].children[1].children[0].own_wait = micros(30);
    pass.breakdown.tree[3].children[0].own_wait = micros(10);

    let rendered = Report::new(&pass);
    let query = &pass.breakdown.tree[1].children[1].children[0];
    // Three quarters of the 710 µs the device waited in the clean pass.
    let host = rendered.host(query).expect("a pass timed in place splits");
    assert!(
        host.abs_diff(Duration::from_nanos(532_500)) < Duration::from_nanos(10),
        "{host:?}"
    );
    assert_eq!(
        rendered.host(&pass.breakdown.tree[1]),
        rendered.host(query),
        "a component's host time is its children's as well"
    );

    let table = rendered.to_string();
    assert!(table.contains("waited"), "{table}");
    assert!(
        table
            .lines()
            .any(|line| line.contains("query") && line.contains("fewer launches")),
        "the worklist names the lever:\n{table}"
    );

    // Timed alone there is no host time to split.
    pass.breakdown.timed_alone = true;
    assert_eq!(Report::new(&pass).host(query), None);
}

/// Timed alone, the kernels' sum is a serialized pass's, so subtracting it
/// from the clean pass would print the lost overlap as host time.
#[test]
fn a_pass_timed_alone_does_not_split_off_the_host() {
    let pass = pass();
    assert_eq!(Report::new(&pass).framework(), None);
    assert_eq!(Report::new(&pass).ladder().host, Duration::ZERO);

    let table = Report::new(&pass).to_string();
    let framework = table
        .lines()
        .find(|line| line.trim_start().starts_with("framework"))
        .expect("the framework line");
    assert!(framework.contains('—'), "{framework}");
}

/// With nothing measured on the device there is no split to make — a pass
/// on a target the runtime does not observe would otherwise read as all
/// host.
#[test]
fn a_pass_with_no_device_time_splits_nothing() {
    let mut pass = pass();
    pass.breakdown.timed_alone = false;
    for component in pass.breakdown.tree.iter_mut() {
        clear(component);
    }
    pass.breakdown.untagged.clear();
    assert_eq!(Report::new(&pass).framework(), None);

    fn clear(component: &mut Component) {
        component.kernels.clear();
        component.children.iter_mut().for_each(clear);
    }
}

/// A composite that prices only the traffic it must move is still held
/// to every multiply-add beneath it: fusion removes traffic, never
/// arithmetic. The whole model declares its floor that way.
#[test]
fn a_composite_is_held_to_the_arithmetic_of_its_parts() {
    let pass = pass();
    let parts: u64 = pass
        .breakdown
        .tree
        .iter()
        .filter_map(|component| component.bounds)
        .map(|bounds| bounds.flops)
        .sum();
    let model = Component {
        label: "transformer".into(),
        bounds: Some(Bounds {
            read: 1_000_000,
            write: 1_000,
            flops: 0,
        }),
        children: pass.breakdown.tree.clone(),
        ..component()
    };

    let held = Report::held(&model).expect("the model declared bounds");
    assert_eq!(held.flops, parts);
    assert_eq!(
        (held.read, held.write),
        (1_000_000, 1_000),
        "its traffic is its own"
    );
}

/// An unpriced root — the model itself — does not hide the priced
/// components under it: the fusion rung still takes each at its own floor
/// rather than falling to the kernel rung.
#[test]
fn an_unpriced_root_keeps_the_fusion_rung() {
    let pass = pass();
    let flat = Report::new(&pass).ladder();

    let mut rooted = pass.clone();
    rooted.breakdown.tree = vec![Component {
        label: "transformer".into(),
        children: std::mem::take(&mut rooted.breakdown.tree),
        ..component()
    }];
    let under_a_root = Report::new(&rooted).ladder();

    assert_eq!(flat.fused, under_a_root.fused);
    assert_eq!(flat.kernels, under_a_root.kernels);
    assert!(under_a_root.from_fusion() > Duration::ZERO);
}

/// The fused work the last components left for the sync ending the pass
/// is printed under its own heading and counted in what the device did.
#[test]
fn the_end_of_the_pass_is_printed_and_counted() {
    let mut pass = pass();
    let before = pass.breakdown.total_time();
    pass.breakdown.end_of_pass = vec![kernel("FusedMatmul", 1, 40)];

    let table = Report::new(&pass).to_string();
    assert!(table.contains("end of pass"), "{table}");
    assert_eq!(
        pass.breakdown.total_time(),
        before + Duration::from_micros(40)
    );
}

#[test]
fn an_unprobed_device_scores_nothing_rather_than_printing_a_guess() {
    let mut pass = pass();
    pass.peaks = None;
    let table = Report::new(&pass).to_string();
    assert!(table.contains("unprobed"), "the header says so:\n{table}");
    // Every roofline column reads as absent rather than as zero.
    assert!(!table.contains('%') || table.contains('—'), "{table}");
}

/// The wgpu reading that prompted the guard: a write ceiling above the
/// device's own read ceiling, on a bus that carries neither. The pass must
/// say the pair was dropped rather than quietly fall back, or a reader
/// cannot tell it from a device that reported no directions at all.
#[test]
fn an_impossible_write_ceiling_is_reported_not_divided_by() {
    let broken = Peaks {
        copy_bytes_per_s: Some(156.0e9),
        read_bytes_per_s: Some(227.0e9),
        write_bytes_per_s: Some(281.0e9),
        flops: Some(ComputePeak {
            per_s: 49.3e12,
            dtype: FloatDType::F16,
        }),
    };
    let pass = CollectedPass {
        peaks: Some(broken),
        ..pass()
    };
    let table = Report::new(&pass).to_string();

    assert!(
        table.contains("REJECTED the per-direction peaks"),
        "the header says so:\n{table}"
    );
    // The rejection names both readings — that is the point of naming it —
    // so what must be absent is the line presenting them as the denominator,
    // not the numbers themselves.
    assert!(
        !table.contains("GB/s read and"),
        "and does not present them as denominators:\n{table}"
    );
    // And it is not also reported as merely disagreeing: one finding, named
    // once, in the terms that say what happened to the denominator.
    assert!(!table.contains("CAVEAT"), "named once:\n{table}");
}

/// Three ceilings that cannot all be measurements of one device are named,
/// and still divided by — nothing in the readings says which is wrong.
#[test]
fn ceilings_that_do_not_predict_each_other_are_a_caveat_not_a_fallback() {
    // Plausible in their ordering, so nothing is rejected; but 234 and 212
    // predict a 222 GB/s copy ceiling against a measured 140.
    let odd = Peaks {
        copy_bytes_per_s: Some(140.0e9),
        read_bytes_per_s: Some(234.0e9),
        write_bytes_per_s: Some(212.0e9),
        flops: Some(ComputePeak {
            per_s: 49.3e12,
            dtype: FloatDType::F16,
        }),
    };
    let pass = CollectedPass {
        peaks: Some(odd),
        ..pass()
    };
    let table = Report::new(&pass).to_string();

    assert!(table.contains("CAVEAT"), "the header says so:\n{table}");
    assert!(
        !table.contains("REJECTED the per-direction peaks"),
        "and nothing was dropped:\n{table}"
    );
    assert!(
        table.contains("234.0 GB/s read"),
        "the components are still scored on the directions:\n{table}"
    );
}

/// The failure a real ROCm run hit: the f16 CMMA probe returned a tiny
/// positive number, so every component read as thousands of percent of peak with
/// a floor longer than the pass. The pass must say the denominator broke
/// rather than render it.
#[test]
fn a_broken_compute_peak_is_reported_not_divided_by() {
    let broken = Peaks {
        copy_bytes_per_s: Some(166.1e9),
        flops: Some(ComputePeak {
            per_s: 4.5e9,
            dtype: FloatDType::F16,
        }),
        ..Peaks::default()
    };
    let pass = CollectedPass {
        peaks: Some(broken),
        ..pass()
    };
    let table = Report::new(&pass).to_string();

    assert!(table.contains("REJECTED"), "the header says so:\n{table}");
    assert!(
        table.contains("unprobed"),
        "and does not print the rejected peak as though it were one:\n{table}"
    );
    // Falls back to the `bytes` wall rather than producing no components at all.
    assert!(table.contains("bytes"), "{table}");
    // Structurally, not by string: every component falls back to the `bytes`
    // wall, since the one it would otherwise be held against has no
    // denominator anyone measured.
    let rendered = Report::new(&pass);
    for (_, component) in pass
        .breakdown
        .tree
        .iter()
        .flat_map(|component| component.walk())
    {
        if let Some(score) = rendered.score_of(component) {
            assert_eq!(score.bound, Bound::Bytes, "{}", component.label);
        }
    }
}

/// A component that beats its floor is at the roof with a probe reading
/// slightly low — the head, streaming 311 MB of weights, does this. It is
/// usually the most interesting component on the page, so it is marked rather
/// than blanked: suppressing its numbers would lose the finding.
///
/// And it keeps its percentage, because how far past is the whole reading:
/// this component is at 118%, which is a probe a fifth low; a component at 568% is a
/// declared bound that is wrong. A shared ">100%" gave both the same
/// answer.
#[test]
fn a_row_past_the_peak_is_marked_and_still_shows_its_numbers() {
    let mut pass = pass();
    // One component whose kernels ran just under its own floor. The roofline is
    // scored on device time, so the component needs a timed kernel rather than a
    // wall time — a component with no kernel has nothing to be a
    // percentage of.
    pass.breakdown.tree.truncate(1);
    pass.breakdown.tree[0].bounds = Some(Bounds {
        read: 240_000_000,
        write: 80_000_000,
        flops: 0,
    });
    pass.breakdown.tree[0].children.clear();
    pass.breakdown.tree[0].kernels = vec![kernel("TheHead", 1, 1_700)];
    pass.breakdown.untagged.clear();
    // The one component is the whole pass, so its apportioned share is the
    // wall — and 1.7 ms against a 2.0 ms floor is a component at the roof.
    pass.clean_wall = Some(Duration::from_micros(1_700));

    let peaks = Peaks {
        copy_bytes_per_s: Some(160.0e9),
        flops: Some(ComputePeak {
            per_s: 32.0e12,
            dtype: FloatDType::F16,
        }),
        ..Peaks::default()
    };
    pass.peaks = Some(peaks);
    let table = Report::new(&pass).to_string();
    // 320 MB against a 160 GB/s ceiling is a 2.0 ms floor, reached in 1.7.
    assert!(
        table.contains("118%*"),
        "the component is marked and keeps its figure:\n{table}"
    );
    assert!(
        table.contains("past the measured peak"),
        "and the mark is explained:\n{table}"
    );
    // The component keeps its numbers. An earlier version blanked such a component as
    // "denominator broken", which on a real run hid the LM head — the best
    // component on the page — behind two dashes.
    let component = table
        .lines()
        .find(|line| line.contains('*'))
        .expect("the marked component");
    assert!(
        !component.contains('—'),
        "a component at the roof keeps its floor and gain:\n{component}"
    );
}

/// A component fed from two places says so, and names them.
///
/// Without the note a merge is invisible: two entry points under one
/// label read exactly like one entered twice. The note leaves the tree
/// alone — merging is what makes a 36-layer stack readable — and points at
/// the sites, which is what a split is made from.
#[test]
fn a_row_fed_from_two_sites_says_where_from() {
    let mut pass = pass();
    pass.breakdown.tree[1].children[1].sites = vec![
        "my_model::attention::base::forward_cached".to_string(),
        "my_model::attention::base::decode_batch".to_string(),
    ];

    let table = Report::new(&pass).to_string();
    assert!(
        table.contains(
            "note: `attention` under `transformer block` was entered from 2 sites \
             (attention::forward_cached, attention::decode_batch)"
        ),
        "the note names the component, its parent and the sites:\n{table}"
    );
}

/// The ordinary component — one site — carries no note. A footnote on every component
/// is a footnote a reader stops reading.
#[test]
fn a_row_fed_from_one_site_carries_no_note() {
    let table = Report::new(&pass()).to_string();
    assert!(
        !table.contains("was entered from"),
        "nothing to say about a component nothing merged into:\n{table}"
    );
}

/// A site is shortened to what a reader greps for: the crate is the same
/// for most components in the tree, and a `base` module holds a folder's one
/// implementation, so neither tells anyone anything.
#[test]
fn a_site_is_named_by_its_module_and_entry() {
    assert_eq!(site("my_model::rope::base::apply"), "rope::apply");
    assert_eq!(
        site("my_model::attention::cache::base::append"),
        "attention::cache::append"
    );
}

/// A component whose execution is chosen at call time says which one ran.
///
/// The attention engine is one label serving four executions — the tile fold,
/// the flash op, tensor ops and the decode kernel. Without the variant they
/// render as the same word, and the component a reader opened the table for reads
/// as though nothing were decided in it.
#[test]
fn a_row_prints_which_executions_served_it() {
    let mut pass = pass();
    pass.breakdown.tree[0].label = "attention engine".to_string();
    pass.breakdown.tree[0].children.clear();
    pass.breakdown.tree[0].entries = 9;
    pass.breakdown.tree[0].variants = vec![
        Variant {
            name: "tile".to_string(),
            entries: 8,
        },
        Variant {
            name: "tensor-ops".to_string(),
            entries: 1,
        },
    ];

    let table = Report::new(&pass).to_string();
    assert!(
        table.contains("attention engine ×9 · tile ×8, tensor-ops ×1"),
        "the split is beside the entry count:\n{table}"
    );
}

/// One execution that served every entry needs no count: it would only
/// repeat the component's own, and the reading — "this component took the decode
/// kernel, all nine times" — is the same either way.
#[test]
fn a_row_served_one_way_prints_the_bare_name() {
    let mut pass = pass();
    pass.breakdown.tree[0].label = "attention engine".to_string();
    pass.breakdown.tree[0].children.clear();
    pass.breakdown.tree[0].entries = 28;
    pass.breakdown.tree[0].variants = vec![Variant {
        name: "decode".to_string(),
        entries: 28,
    }];

    let table = Report::new(&pass).to_string();
    assert!(
        table.contains("attention engine ×28 · decode"),
        "the name alone:\n{table}"
    );
    assert!(
        !table.contains("decode ×28"),
        "and not its count twice:\n{table}"
    );
}

/// The component column sizes itself to what it has to hold, so a long
/// variant widens the table rather than pushing one component's numbers out of
/// their columns and leaving the rest aligned to nothing.
#[test]
fn every_row_keeps_its_columns_when_a_heading_grows() {
    let mut pass = pass();
    pass.breakdown.tree[0].label = "attention engine".to_string();
    pass.breakdown.tree[0].entries = 9;
    pass.breakdown.tree[0].variants = vec![
        Variant {
            name: "prefill-fold".to_string(),
            entries: 8,
        },
        Variant {
            name: "tensor-ops".to_string(),
            entries: 1,
        },
    ];

    let table = Report::new(&pass).to_string();
    // Where a right-aligned column ends, on the header and on the
    // subheading beneath it.
    let ends = |line: &str, word: &str| {
        table
            .lines()
            .find(|found| found.contains(line))
            .and_then(|found| found.find(word))
            .map(|at| at + word.len())
    };
    assert_eq!(
        ends("gain ms", "share"),
        ends("recoverable", "of work"),
        "the header and its subheading share one column layout:\n{table}"
    );
    // And the rule follows the column rather than staying at the width a
    // narrower table used, which would leave it ending mid-heading.
    let rule = table
        .lines()
        .map(|line| line.trim())
        .find(|line| !line.is_empty() && line.chars().all(|glyph| glyph == '─'))
        .expect("the rule under the header")
        .chars()
        .count();
    assert!(
        rule > 88,
        "the rule widened with the column ({rule} columns):\n{table}"
    );
    // And the heading ends inside its column rather than running into the
    // figures — which is the failure the fixed width would have produced.
    let widened = table
        .lines()
        .find(|line| line.contains("attention engine"))
        .expect("the widened component");
    let after = widened
        .split_once("tensor-ops ×1")
        .expect("the heading, in full")
        .1;
    assert!(
        after.starts_with(' '),
        "the figures do not butt up against the heading:\n{widened}"
    );
}

/// A kernel's figures sit in its component's columns, whatever the depth:
/// a kernel line two columns to the right reads its milliseconds as the
/// share beside them.
#[test]
fn a_kernel_line_keeps_its_components_columns() {
    let pass = pass();
    let table = Report::new(&pass).to_string();
    // The column, in characters, where the milliseconds figure after the
    // label ends: `·` and `×` are wider than a byte.
    let ms_end = |needle: &str| {
        let line = table
            .lines()
            .find(|line| line.contains(needle))
            .unwrap_or_else(|| panic!("no line with {needle}:\n{table}"));
        let (before, after) = line.split_once(needle).expect("found above");
        let digits = after
            .find(|glyph: char| glyph.is_ascii_digit())
            .expect("a figure");
        let figure = after[digits..].split(' ').next().expect("a figure");
        before.chars().count()
            + needle.chars().count()
            + after[..digits].chars().count()
            + figure.chars().count()
    };
    for (component, kernel) in [
        ("embedding", "· Gather ×1"),
        ("query ×2", "· Matmul ×2"),
        ("attn norm ×2", "· RmsNormKernel ×2"),
    ] {
        assert_eq!(ms_end(component), ms_end(kernel), "{kernel}:\n{table}");
    }
    assert_eq!(
        ms_end("embedding"),
        ms_end("· Reshape ×1"),
        "an untagged kernel too:\n{table}"
    );
}

/// Past the column's cap a heading is cut, and cut visibly.
///
/// The cap exists so one name cannot push every number off a terminal; the
/// component it caps still has to line up with the rest, because the component with the
/// most to say — the one that routed four ways — is the one a reader came
/// for, and it would be the only unreadable line in the table.
#[test]
fn a_heading_past_the_cap_is_elided_rather_than_overflowing() {
    let mut pass = pass();
    pass.breakdown.tree[0].label = "an operator with a very long name indeed".to_string();
    pass.breakdown.tree[0].children.clear();
    pass.breakdown.tree[0].entries = 4;
    pass.breakdown.tree[0].variants = ["tile", "cubek", "prefill-fold", "tensor-ops"]
        .into_iter()
        .map(|name| Variant {
            name: name.to_string(),
            entries: 1,
        })
        .collect();

    let table = Report::new(&pass).to_string();
    let component = table
        .lines()
        .find(|line| line.contains("an operator with"))
        .expect("the capped component");
    assert!(
        component.contains('…'),
        "the cut is marked rather than silent:\n{component}"
    );

    // And its figures sit in the columns the header set, which is the
    // property the cut protects. Measured in columns rather than bytes:
    // `×` and `…` are wider than one byte, and an offset would make the
    // elided component look misaligned when it is not.
    let header = table
        .lines()
        .find(|line| line.contains("component") && line.contains("gain ms"))
        .expect("the header");
    // The bare `ms` column, right-aligned like every value under it.
    let at = header.find(" ms").expect("the ms column") + 3;
    let column = header[..at].chars().count();

    let ordinary = table
        .lines()
        .find(|line| line.contains("final norm"))
        .expect("a component with a short heading");
    for line in [component, ordinary] {
        let glyph = |column: usize| line.chars().nth(column);
        assert!(
            glyph(column - 1).is_some_and(|glyph| glyph.is_ascii_digit()),
            "a figure ends where the header's `ms` does:\n{line}"
        );
        assert_eq!(
            glyph(column),
            Some(' '),
            "and the next column starts after it:\n{line}"
        );
    }
}

/// Saturating the ladder's subtractions would print "0.0 on the table"
/// under rungs that plainly exceed the pass — which reads as "nothing to
/// win", the opposite of what happened.
#[test]
fn an_incoherent_ladder_says_so_rather_than_reporting_nothing_to_win() {
    let inverted = Ladder {
        measured: Duration::from_millis(715),
        host: Duration::ZERO,
        kernels: Duration::from_millis(88_535),
        fused: Duration::from_millis(102_582),
    };
    assert!(!inverted.coherent());
    assert_eq!(
        inverted.total(),
        Duration::ZERO,
        "the saturation is why this must not be printed as a result"
    );

    let real = Ladder {
        measured: Duration::from_millis(1000),
        host: Duration::ZERO,
        kernels: Duration::from_millis(600),
        fused: Duration::from_millis(400),
    };
    assert!(real.coherent());
}

/// A priced component that measured no time keeps its bound and floor: its
/// parent's floor counts them, and blanks would leave that figure with
/// nothing to be reconciled against.
#[test]
fn an_unscored_priced_row_keeps_its_floor() {
    let mut pass = pass();
    pass.breakdown.tree[1].children[1].children[0].kernels[0].total = Duration::ZERO;

    let rendered = Report::new(&pass);
    let query = &pass.breakdown.tree[1].children[1].children[0];
    assert!(
        rendered.score_of(query).is_none(),
        "nothing measured to score"
    );
    let floor = rendered.floor_of(query).expect("the query is priced");
    let table = rendered.to_string();
    let component = table
        .lines()
        .find(|line| line.trim_start().starts_with("query"))
        .expect("the query's component");
    assert!(component.contains(&floor.bound.to_string()), "{component}");
    assert!(component.contains(&ms(floor.time)), "{component}");
}

/// A kernel that launched and measured nothing is a lost timing, said in
/// the summary — and the marks and the worklist that read those timings
/// say they cannot be trusted rather than "nothing to win".
#[test]
fn a_kernel_that_measured_no_time_is_reported_lost() {
    let mut pass = pass();
    assert_eq!(pass.breakdown.untimed().count(), 0);
    let table = Report::new(&pass).to_string();
    assert!(!table.contains("untimed"), "{table}");

    pass.breakdown.tree[1].children[1].children[0].kernels[0].total = Duration::ZERO;
    assert_eq!(pass.breakdown.untimed().count(), 1);
    let table = Report::new(&pass).to_string();
    assert!(
        table.contains("untimed") && table.contains("1 kernels, 2 launches"),
        "{table}"
    );
    assert!(!table.contains("Nothing to win"), "{table}");
}

/// A wait before a single launch is the host's work ahead of it; no
/// launch can be taken away from one, so it is not "fewer launches".
#[test]
fn a_wait_before_one_launch_is_host_work() {
    let mut pass = pass();
    pass.breakdown.timed_alone = false;
    pass.breakdown.tree[0].own_wait = Duration::from_micros(30);

    let table = Report::new(&pass).to_string();
    let line = table
        .lines()
        .find(|line| line.contains("embedding") && line.contains("host work"))
        .unwrap_or_else(|| panic!("the embedding's worklist line:\n{table}"));
    assert!(line.contains("before its one launch"), "{line}");
}

#[test]
fn the_ladder_never_promises_more_than_the_pass_took() {
    let pass = pass();
    let rendered = Report::new(&pass);
    let ladder = rendered.ladder();
    assert!(
        ladder.total() <= ladder.measured,
        "a gain larger than the time it came from is not a gain"
    );
    assert!(
        ladder.fused <= ladder.kernels,
        "perfect fusion cannot be slower than each part at its own roof"
    );
}

#[test]
fn a_repeated_component_says_how_many_times_it_ran() {
    let pass = pass();
    let table = Report::new(&pass).to_string();
    // Two layers, one component: without the count a reader would divide by the
    // wrong number to get a per-layer figure.
    assert!(
        table.contains("transformer block ×2"),
        "the stack's component carries its layer count:\n{table}"
    );
}
