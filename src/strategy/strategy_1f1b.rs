use std::cell::RefCell;

use anyhow::{anyhow, Ok, Result};

use crate::strategy::Strategy;

use crate::minibatch::MiniBatch;
use crate::stage::Stage;

#[derive(Debug, Clone)]
struct StageState {
    stage: Stage,
    forward_idx: usize,
    backward_idx: usize,
    curr_time: usize,
    warm_up: usize,
    steady: usize,
    cool_down: usize,
}

impl StageState {
    fn new(stage_idx: usize) -> Self {
        StageState {
            stage: Stage::new(stage_idx),
            forward_idx: 0,
            backward_idx: 0,
            curr_time: 0,
            warm_up: 0,
            steady: 0,
            cool_down: 0,
        }
    }

    fn complete(&self, num_minibatch: usize) -> bool {
        self.forward_idx >= num_minibatch && self.backward_idx >= num_minibatch
    }
}

pub struct Strategy1F1B {
    stage_states: Vec<RefCell<StageState>>,
    world_size: usize,
    num_minibatch: usize,
    arrangements: Option<Vec<Vec<MiniBatch>>>,
}

impl Strategy1F1B {
    fn init_1f1b_devices(world_size: usize, num_minibatch: usize) -> Vec<RefCell<StageState>> {
        let mut devices: Vec<RefCell<StageState>> = (0..world_size).map(
            |rank   | {RefCell::new(StageState::new(rank))}
        ).collect();
        for (stage_idx, state) in devices.iter_mut().enumerate() {
            let mut state = state.borrow_mut();
            state.stage.prev_stage = if stage_idx == 0 { None } else { Some(stage_idx - 1) };
            state.stage.next_stage = if stage_idx == world_size - 1 {
                None
            } else {
                Some(stage_idx + 1)
            };
            state.warm_up = world_size - stage_idx;
            state.steady = num_minibatch - state.warm_up;
            state.cool_down = num_minibatch - state.steady;
        }
        devices
    }
}

impl Strategy for Strategy1F1B {
    fn new(world_size: usize, num_minibatch: usize) -> Self {
        let devices = Self::init_1f1b_devices(world_size, num_minibatch);
        let arrangements = Some(vec![Vec::<MiniBatch>::new(); world_size]);
        Self {
            stage_states: devices,
            world_size,
            num_minibatch,
            arrangements,
        }
    }

    fn complete(&self) -> bool {
        self.stage_states.iter().all(
            |s| {
                s.borrow().complete(self.num_minibatch)
            }
        )
    }

    fn step(&mut self) -> Result<()> {
        let arrangements = self.arrangements
        .as_mut()
        .ok_or(anyhow!("Arrangements not initialized"))?;

        for (stage_idx, state) in self.stage_states.iter_mut().enumerate() {
            let mut state = state.borrow_mut();
            let arrange = &mut arrangements[stage_idx];

            if state.forward_idx < state.warm_up {
                // warmup: n F
                if state.curr_time < stage_idx {
                    arrange.push(MiniBatch::Nops);
                } else {
                    let forward_idx = state.forward_idx;
                    state.forward_idx += 1;
                    arrange.push(MiniBatch::Forward(forward_idx));
                }
            } else if state.forward_idx < self.num_minibatch {
                // steady: 1B1F
                if state.curr_time < self.world_size * 2 - 1 - stage_idx {
                    arrange.push(MiniBatch::Nops);
                } else {
                    let backward_idx = state.backward_idx;
                    state.backward_idx += 1;
                    let forward_idx = state.forward_idx;
                    state.forward_idx += 1;
                    arrange.extend([MiniBatch::Backward(backward_idx), MiniBatch::Forward(forward_idx)]);
                }
            } else if state.backward_idx < self.num_minibatch {
                // cooldown: n B
                let backward_idx = state.backward_idx;
                state.backward_idx += 1;
                arrange.push(MiniBatch::Backward(backward_idx));
                if state.backward_idx < self.num_minibatch {
                    arrange.push(MiniBatch::Nops);
                }
            } else {
                arrange.push(MiniBatch::Nops);
            }

            state.curr_time += 1;
        }
        Ok(())
    }

    fn arrangements(&mut self) -> Result<Vec<Vec<MiniBatch>>> {
        self.arrangements
            .take()
            .ok_or(anyhow!("Arrangements not initialized"))
    }
}
