use super::*;

/// A plain node for exercising the graph algorithms without any tensor machinery.
#[derive(Default)]
struct TestNode {
    position: usize,
    produced: Vec<u32>,
    read: Vec<u32>,
    freed: Vec<u32>,
}

impl TestNode {
    fn new(position: usize) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    fn produces(mut self, resources: impl IntoIterator<Item = u32>) -> Self {
        self.produced.extend(resources);
        self
    }

    fn reads(mut self, resources: impl IntoIterator<Item = u32>) -> Self {
        self.read.extend(resources);
        self
    }

    /// Reads the resources for the last time (both read and freed).
    fn frees(mut self, resources: impl IntoIterator<Item = u32>) -> Self {
        for resource in resources {
            self.read.push(resource);
            self.freed.push(resource);
        }
        self
    }
}

impl GraphNode for TestNode {
    type Resource = u32;

    fn produced(&self) -> impl Iterator<Item = u32> {
        self.produced.iter().copied()
    }

    fn read(&self) -> impl Iterator<Item = u32> {
        self.read.iter().copied()
    }

    fn freed(&self) -> impl Iterator<Item = u32> {
        self.freed.iter().copied()
    }

    fn position(&self) -> usize {
        self.position
    }
}

mod subgraph {
    use super::*;

    #[test]
    fn insert_contains_iter() {
        let mut set = SubGraph::empty();
        assert!(set.is_empty());

        set.insert(3);
        set.insert(70); // Second word.
        assert!(set.contains(3));
        assert!(set.contains(70));
        assert!(!set.contains(4));
        assert_eq!(set.iter().collect::<Vec<_>>(), vec![3, 70]);
    }

    #[test]
    fn set_operations_keep_canonical_form() {
        let mut a = SubGraph::single(2);
        a.insert(100);

        // Subtracting the high bit must trim the trailing word so equality stays structural.
        a.subtract(&SubGraph::single(100));
        assert_eq!(a, SubGraph::single(2));

        let mut b = SubGraph::single(2);
        b.union_with(&SubGraph::single(65));
        assert!(b.intersects(&SubGraph::single(65)));
        assert!(!b.intersects(&SubGraph::single(64)));

        b.intersect_with(&SubGraph::single(2));
        assert_eq!(b, SubGraph::single(2));
    }
}

mod dag {
    use super::*;

    #[test]
    fn independent_nodes_ordered_by_position() {
        let nodes = vec![
            TestNode::new(2).reads([200, 201]).produces([202]),
            TestNode::new(0).reads([100, 101]).produces([102]),
        ];
        let dag = Dag::new(&nodes);

        assert!(!dag.depends_on(0, 1));
        assert!(!dag.depends_on(1, 0));
        assert_eq!(dag.topological_order(), Some(vec![1, 0]));
    }

    #[test]
    fn read_after_write_beats_position() {
        // Node 0 (position 0) reads resource 50, which node 1 (position 5) produces.
        let nodes = vec![
            TestNode::new(0).reads([50]).produces([2]),
            TestNode::new(5).reads([9]).produces([50]),
        ];
        let dag = Dag::new(&nodes);

        assert!(dag.depends_on(0, 1));
        assert_eq!(dag.topological_order(), Some(vec![1, 0]));
    }

    #[test]
    fn mutual_dependency_is_a_cycle() {
        // Node 0 produces 100 (read by node 1) and reads 200 (produced by node 1).
        let nodes = vec![
            TestNode::new(0).reads([200]).produces([100]),
            TestNode::new(1).reads([100]).produces([200]),
        ];
        let dag = Dag::new(&nodes);

        assert!(!dag.is_acyclic());
        assert_eq!(dag.topological_order(), None);
    }

    #[test]
    fn write_after_read_orders_freer_after_reader() {
        // Both nodes read resource 100 (produced elsewhere); node 1 frees it. The freer must
        // run after the reader, even though nothing flows between them.
        let nodes = vec![
            TestNode::new(0).reads([100]).produces([2]),
            TestNode::new(1).frees([100]).produces([4]),
        ];
        let dag = Dag::new(&nodes);

        assert!(dag.depends_on(1, 0));
        assert!(!dag.depends_on(0, 1));
        assert_eq!(dag.topological_order(), Some(vec![0, 1]));
    }

    #[test]
    fn diamond_topological_order() {
        //   0
        //  / \
        // 1   2
        //  \ /
        //   3
        let nodes = vec![
            TestNode::new(0).produces([10]),
            TestNode::new(1).reads([10]).produces([11]),
            TestNode::new(2).reads([10]).produces([12]),
            TestNode::new(3).reads([11, 12]).produces([13]),
        ];
        let dag = Dag::new(&nodes);

        assert_eq!(dag.topological_order(), Some(vec![0, 1, 2, 3]));
    }
}

mod reachability {
    use super::*;

    /// A → C → B chain: contracting A and B would trap C between them.
    fn chain() -> Vec<TestNode> {
        vec![
            TestNode::new(0).produces([100]),            // A
            TestNode::new(1).reads([100]).produces([200]), // C
            TestNode::new(2).reads([200]).produces([300]), // B
        ]
    }

    #[test]
    fn direct_edge_can_contract() {
        let reachability = Dag::new(&chain()).reachability();

        assert!(reachability.can_contract(&SubGraph::single(0), &SubGraph::single(1)));
        assert!(reachability.can_contract(&SubGraph::single(1), &SubGraph::single(2)));
    }

    #[test]
    fn intermediate_node_blocks_contraction() {
        let reachability = Dag::new(&chain()).reachability();

        assert!(!reachability.can_contract(&SubGraph::single(0), &SubGraph::single(2)));
    }

    #[test]
    fn contraction_including_the_intermediate_is_legal() {
        let reachability = Dag::new(&chain()).reachability();

        // Contracting {A, C} with {B} is fine: nothing remains between them.
        let mut a_and_c = SubGraph::single(0);
        a_and_c.union_with(&SubGraph::single(1));
        assert!(reachability.can_contract(&a_and_c, &SubGraph::single(2)));
    }

    #[test]
    fn empty_subgraphs_are_refused() {
        let reachability = Dag::new(&chain()).reachability();

        assert!(!reachability.can_contract(&SubGraph::empty(), &SubGraph::empty()));
        assert!(!reachability.can_contract(&SubGraph::single(0), &SubGraph::empty()));
    }

    #[test]
    fn transitive_ancestors_block_contraction() {
        // 0 → 1 → 2 → 3: node 0 and node 3 have no direct edge, but both 1 and 2 sit between.
        let nodes = vec![
            TestNode::new(0).produces([10]),
            TestNode::new(1).reads([10]).produces([11]),
            TestNode::new(2).reads([11]).produces([12]),
            TestNode::new(3).reads([12]).produces([13]),
        ];
        let reachability = Dag::new(&nodes).reachability();

        assert!(!reachability.can_contract(&SubGraph::single(0), &SubGraph::single(3)));
        assert!(!reachability.can_contract(&SubGraph::single(0), &SubGraph::single(2)));
    }

    #[test]
    fn supports_more_than_64_nodes() {
        // A chain of 70 nodes: node i reads what node i-1 produces.
        let nodes = (0..70)
            .map(|i| {
                let node = TestNode::new(i).produces([i as u32 + 1]);
                match i {
                    0 => node,
                    _ => node.reads([i as u32]),
                }
            })
            .collect::<Vec<_>>();
        let dag = Dag::new(&nodes);
        let reachability = dag.reachability();

        assert_eq!(dag.topological_order(), Some((0..70).collect::<Vec<_>>()));
        assert!(reachability.can_contract(&SubGraph::single(68), &SubGraph::single(69)));
        assert!(!reachability.can_contract(&SubGraph::single(0), &SubGraph::single(69)));
    }
}

mod lifetime {
    use super::*;

    #[test]
    fn program_order_is_valid() {
        let nodes = vec![
            TestNode::new(0).reads([1]).produces([10]),
            TestNode::new(1).frees([10]).produces([11]),
        ];

        assert!(is_valid_execution_order(&nodes, &[0, 1]));
    }

    #[test]
    fn read_before_produce_is_invalid() {
        let nodes = vec![
            TestNode::new(0).reads([1]).produces([10]),
            TestNode::new(1).reads([10]).produces([11]),
        ];

        assert!(!is_valid_execution_order(&nodes, &[1, 0]));
    }

    #[test]
    fn external_reads_are_assumed_live() {
        // Resource 1 is produced by no ordered node: it comes from an earlier segment.
        let nodes = vec![
            TestNode::new(0).reads([1]).produces([10]),
            TestNode::new(1).reads([1, 10]).produces([11]),
        ];

        assert!(is_valid_execution_order(&nodes, &[0, 1]));
    }

    #[test]
    fn order_may_cover_a_subset_of_nodes() {
        // Only node 1 is ordered; the resource node 0 produces counts as external.
        let nodes = vec![
            TestNode::new(0).produces([10]),
            TestNode::new(1).reads([10]).produces([11]),
        ];

        assert!(is_valid_execution_order(&nodes, &[1]));
    }
}
