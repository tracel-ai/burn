#[burn_tensor_testgen::testgen(multi_threads)]
mod tests {
    use super::*;
    use burn_tensor::{DType, Element, Shape, backend::Backend};
    use core::time::Duration;
    use std::sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    };

    #[test]
    fn should_handle_multi_threads() {
        let tensor = TestTensor::<2>::from([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]]);

        let num_threads = 2;
        let num_fused = 3;
        let mut joined = Vec::with_capacity(num_threads);

        let num_poped = Arc::new(AtomicU32::new(0));

        for i in 0..num_threads {
            let tensor_moved = tensor.clone();
            let num_poped_moved = num_poped.clone();

            let handle = std::thread::spawn(move || {
                let mut base = tensor_moved + i as i32;

                num_poped_moved.fetch_add(1, Ordering::Relaxed);

                println!("Poped.");
                for n in 0..num_fused {
                    base = base + n as i32;
                    std::thread::sleep(Duration::from_secs(1));
                }
                let _data = base.into_data();
                println!("Done.");
            });
            joined.push(handle);
        }

        loop {
            let num_poped_current = num_poped.load(Ordering::Relaxed);
            if num_poped_current as usize == num_threads {
                break;
            } else {
                std::thread::sleep(Duration::from_secs(1));
            }
        }
        println!("Dropping");

        let t = tensor * 2.0;
        let _t = t.into_data();
        //core::mem::drop(tensor);

        for j in joined {
            j.join().unwrap();
        }
    }
}
