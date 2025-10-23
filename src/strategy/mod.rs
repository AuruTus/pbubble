use anyhow::{Result, anyhow};

use crate::minibatch::MiniBatch;

mod strategy_1f1b;

pub trait Strategy {
    fn new(world_size: usize, num_minibatch: usize) -> Self;
    fn complete(&self) -> bool;
    fn step(&mut self) -> Result<()>;
    fn arrangements(&mut self) -> Result<Vec<Vec<MiniBatch>>>;
}
