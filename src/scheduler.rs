use anyhow::Result;

use crate::minibatch::MiniBatch;
use crate::strategy::Strategy;

pub struct Scheduler {
    world_size: usize,
}

impl Scheduler {
    pub fn new(world_size: usize) -> Self {
        Scheduler { world_size }
    }

    pub fn run<St: Strategy>(&mut self, num_minibatch: usize) -> Result<Vec<Vec<MiniBatch>>> {
        let mut st = St::new(self.world_size, num_minibatch);

        while !st.complete() {
            st.step()?;
        }

        st.arrangements()
    }
}
