use anyhow::{Ok, Result};
use pbubble::scheduler::Scheduler;
use pbubble::strategy::strategy_1f1b::Strategy1F1B;
use pbubble::utils::print_arrangements_matrix;

#[test]
fn test_1f1b() -> Result<()> {
    let mut scheduler = Scheduler::new(4);
    let arrangements = scheduler.run::<Strategy1F1B>(8)?;

    print_arrangements_matrix(arrangements.as_slice());

    Ok(())
}
