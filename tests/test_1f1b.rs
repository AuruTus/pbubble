use anyhow::{Ok, Result};
use pbubble::scheduler::Scheduler;
use pbubble::strategy::strategy_1f1b::Strategy1F1B;
use pbubble::utils::print_arrangements_matrix;

#[test]
fn test_1f1b() -> Result<()> {
    let test_cases = vec![
        (2, 4),
        (4, 8),
        (4, 10),
        (4, 6),
        (4, 16),
    ];
    for (world_size, num_minibatch) in test_cases {
        let split_mark = "=".repeat(20);

        println!("{} world_size: {}, num_minibatch: {} {}", split_mark, world_size, num_minibatch, split_mark);
        let mut scheduler = Scheduler::new(world_size);
        let arrangements = scheduler.run::<Strategy1F1B>(num_minibatch)?;
        print_arrangements_matrix(arrangements.as_slice());
        println!("\n");
    }

    Ok(())
}
