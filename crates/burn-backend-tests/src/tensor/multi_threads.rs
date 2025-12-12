use super::*;
use core::time::Duration;
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

struct MultiThreadTestSettings {
    num_threads: usize,
    // The number of operations that are applied while the tensor is still alive and has a
    // reference count > 1 on the new thread.
    num_ops_alive: usize,
    // The number of operations that are applied after the tensor is consumed for the last time.
    num_ops_consumed: usize,
    // Number of operations that needs to execute before continuing execution on the main thread.
    sleep_before: Duration,
    sleep_alive: Duration,
    sleep_consumed: Duration,
    // If the output is dropped, otherwise it will be consumed by an operation.
    dropped: bool,
}

#[test]
fn should_handle_multi_threads_dropped() {
    run_multi_thread_test(MultiThreadTestSettings {
        num_threads: 3,
        num_ops_alive: 5,
        num_ops_consumed: 5,
        sleep_before: Duration::from_millis(100),
        sleep_alive: Duration::from_millis(100),
        sleep_consumed: Duration::from_millis(100),
        dropped: true,
    })
}

#[test]
fn should_handle_multi_threads_consumed() {
    run_multi_thread_test(MultiThreadTestSettings {
        num_threads: 3,
        num_ops_alive: 5,
        num_ops_consumed: 5,
        sleep_before: Duration::from_millis(100),
        sleep_alive: Duration::from_millis(100),
        sleep_consumed: Duration::from_millis(100),
        dropped: false,
    })
}

#[test]
fn should_handle_multi_threads_drop_no_wait() {
    run_multi_thread_test(MultiThreadTestSettings {
        num_threads: 3,
        num_ops_alive: 5,
        num_ops_consumed: 5,
        sleep_before: Duration::from_millis(100),
        sleep_alive: Duration::from_millis(100),
        sleep_consumed: Duration::from_millis(100),
        dropped: true,
    })
}

#[test]
fn should_handle_multi_threads_consumed_no_wait() {
    run_multi_thread_test(MultiThreadTestSettings {
        num_threads: 3,
        num_ops_alive: 5,
        num_ops_consumed: 5,
        sleep_before: Duration::from_millis(100),
        sleep_alive: Duration::from_millis(100),
        sleep_consumed: Duration::from_millis(100),
        dropped: false,
    })
}

#[test]
fn should_handle_multi_threads_no_async_op() {
    run_multi_thread_test(MultiThreadTestSettings {
        num_threads: 3,
        num_ops_alive: 0,
        num_ops_consumed: 0,
        sleep_before: Duration::from_millis(100),
        sleep_alive: Duration::from_millis(100),
        sleep_consumed: Duration::from_millis(100),
        dropped: false,
    })
}

#[test]
fn should_handle_multi_threads_no_async_op_no_wait() {
    run_multi_thread_test(MultiThreadTestSettings {
        num_threads: 3,
        num_ops_alive: 0,
        num_ops_consumed: 0,
        sleep_before: Duration::from_millis(0),
        sleep_alive: Duration::from_millis(100),
        sleep_consumed: Duration::from_millis(100),
        dropped: false,
    })
}

fn run_multi_thread_test(settings: MultiThreadTestSettings) {
    let tensor = TestTensor::<2>::from([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]]);

    let mut joined = Vec::with_capacity(settings.num_threads);

    let counter_alive = Arc::new(AtomicU32::new(0));
    let counter_consumed = Arc::new(AtomicU32::new(0));

    for i in 0..settings.num_threads {
        let tensor_moved = tensor.clone();
        let ca_moved = counter_alive.clone();
        let cc_moved = counter_consumed.clone();

        let handle = std::thread::spawn(move || {
            let mut base = tensor_moved.clone();
            std::thread::sleep(settings.sleep_before);

            if settings.num_ops_alive == 0 && settings.num_ops_consumed == 0 {
                core::mem::drop(tensor_moved);
                core::mem::drop(base);
            } else {
                if settings.num_ops_alive > 1 {
                    for j in 0..(settings.num_ops_alive - 1) {
                        base = tensor_moved.clone() + j as u32;
                        ca_moved.fetch_add(1, Ordering::Relaxed);
                        std::thread::sleep(settings.sleep_alive);
                    }
                }

                base = base * tensor_moved + i as u32;
                ca_moved.fetch_add(1, Ordering::Relaxed);

                for n in 0..settings.num_ops_consumed {
                    base = base + n as i32;
                    cc_moved.fetch_add(1, Ordering::Relaxed);
                    std::thread::sleep(settings.sleep_consumed);
                }
                let _data = base.into_data();
            }
        });
        joined.push(handle);
    }

    fn wait(counter: Arc<AtomicU32>, limit: usize) {
        loop {
            let counter_curr = counter.load(Ordering::Relaxed);
            if counter_curr as usize >= limit {
                break;
            } else {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    wait(counter_alive, settings.num_ops_alive);
    wait(counter_consumed, settings.num_ops_consumed);

    if settings.dropped {
        core::mem::drop(tensor);
    } else {
        let t = tensor * 2.0;
        let _t = t.into_data();
    }

    for j in joined {
        j.join().unwrap();
    }
}
