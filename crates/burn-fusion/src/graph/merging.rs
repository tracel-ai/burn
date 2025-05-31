use super::{Graph, GraphMergingResult};
use crate::NumOperations;

pub fn merge_graphs<O: NumOperations>(graphs: &[&Graph<O>]) -> Option<Graph<O>> {
    let combinations = generate_combinations(graphs.len(), graphs.len());

    for mut combination in combinations.into_iter() {
        let first_index = combination.remove(0);
        let mut result = graphs[first_index].clone();
        let mut working = true;

        for i in combination {
            match result.merge(&graphs[i]) {
                GraphMergingResult::Fail => {
                    working = false;
                    break;
                }
                GraphMergingResult::Succeed => {}
            }
        }

        if working {
            return Some(result);
        }
    }

    None
}

fn generate_combinations(digits: usize, n: usize) -> Vec<Vec<usize>> {
    // TODO: Optimize that.
    let total = n.pow(digits as u32);
    let mut result = Vec::with_capacity(total);

    for i in 0..total {
        let mut combination = Vec::with_capacity(digits);
        let mut num = i;
        for _ in 0..digits {
            combination.push(num % n);
            num /= n;
        }
        combination.reverse();

        let mut seen = vec![false; n];
        let mut unique = true;
        for &x in &combination {
            if seen[x] {
                unique = false;
                break;
            }
            seen[x] = true;
        }
        if unique {
            result.push(combination);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_combinations_test() {
        let values = generate_combinations(3, 3);
        panic!("{values:?}");
    }
}
