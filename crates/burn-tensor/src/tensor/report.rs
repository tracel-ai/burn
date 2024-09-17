use super::{backend::Backend, Tensor};

use colored::*;

/// Checks the closeness of two tensors and prints the results.
///
/// Compares tensors by checking the absolute difference between each element.
/// Prints the percentage of elements within specified tolerances.
///
/// # Arguments
///
/// * `output` - The output tensor.
/// * `expected` - The expected tensor.
///
/// # Example
///
/// ```no_run
/// use burn_tensor::backend::Backend;
/// use burn_tensor::{check_closeness, Tensor};
///
/// fn example<B: Backend>() {
///     let device = Default::default();
///     let tensor1 = Tensor::<B, 1>::from_floats(
///         [1.0, 2.0, 3.0, 4.0, 5.0, 6.001, 7.002, 8.003, 9.004, 10.1],
///         &device,
///     );
///     let tensor2 = Tensor::<B, 1>::from_floats(
///         [1.0, 2.0, 3.0, 4.000, 5.0, 6.0, 7.001, 8.002, 9.003, 10.004],
///         &device,
///     );
///    check_closeness(&tensor1, &tensor2);
///}
/// ```
///
/// # Output
///
/// ```text
/// Tensor Closeness Check Results:
/// ===============================
/// Epsilon: 1e-1
///   Close elements: 10/10 (100.00%)
///   [PASS] All elements are within tolerance
///
/// Epsilon: 1e-2
///   Close elements: 10/10 (100.00%)
///   [PASS] All elements are within tolerance
///
/// Epsilon: 1e-3
///   Close elements: 9/10 (90.00%)
///   [WARN] Most elements are within tolerance
///
/// Epsilon: 1e-4
///   Close elements: 6/10 (60.00%)
///   [FAIL] Significant differences detected
///
/// Epsilon: 1e-5
///   Close elements: 5/10 (50.00%)
///   [FAIL] Significant differences detected
///
/// Epsilon: 1e-6
///   Close elements: 5/10 (50.00%)
///   [FAIL] Significant differences detected
///
/// Epsilon: 1e-7
///   Close elements: 5/10 (50.00%)
///   [FAIL] Significant differences detected
///
/// Epsilon: 1e-8
///   Close elements: 5/10 (50.00%)
///   [FAIL] Significant differences detected
///
/// Closeness check complete.
/// ```

pub fn check_closeness<B: Backend, const D: usize>(output: &Tensor<B, D>, expected: &Tensor<B, D>) {
    println!("{}", "Tensor Closeness Check Results:".bold());
    println!("===============================");

    for epsilon in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8].iter() {
        println!("{} {:.e}", "Epsilon:".bold(), epsilon);

        let close = output
            .clone()
            .is_close(expected.clone(), Some(*epsilon), Some(*epsilon));
        let data = close.clone().into_data();
        let num_elements = data.num_elements();

        // Count the number of elements that are close (true)
        let count = data.iter::<bool>().filter(|x| *x).count();

        let percentage = (count as f64 / num_elements as f64) * 100.0;

        println!(
            "  Close elements: {}/{} ({:.2}%)",
            count, num_elements, percentage
        );

        if percentage == 100.0 {
            println!("  {} All elements are within tolerance", "[PASS]".green());
        } else if percentage >= 90.0 {
            println!("  {} Most elements are within tolerance", "[WARN]".yellow());
        } else {
            println!("  {} Significant differences detected", "[FAIL]".red());
        }

        println!();
    }

    println!("{}", "Closeness check complete.".bold());
}
