use std::cell::RefCell;

use anyhow::{Ok, Result, anyhow};

use crate::strategy::{self, Strategy};

use crate::minibatch::MiniBatch;
use crate::stage::Stage;

#[derive(Debug)]
struct StageState {
    device: Stage,
    forward_idx: usize,
    backward_idx: usize,
    curr_time: usize,
}

impl StageState {
    fn new(stage_idx: usize) -> Self {
        StageState {
            device: Stage::new(stage_idx),
            forward_idx: 0,
            backward_idx: 0,
            curr_time: 0,
        }
    }

    fn complete(&self, num_minibatch: usize) -> bool {
        self.forward_idx >= num_minibatch && self.backward_idx >= num_minibatch
    }
}

pub struct Strategy1F1B {
    stage_states: Vec<StageState>,
    world_size: usize,
    num_minibatch: usize,
    arrangements: Option<Vec<Vec<MiniBatch>>>,
}

struct StatesGuard<'a> {
    stage_states: Vec<StageState>,
    strategy: &'a mut Strategy1F1B,
}

impl<'a> StatesGuard<'a> {
    fn guard(strategy: &'a mut Strategy1F1B) -> Self {
        let stage_states = std::mem::take(&mut strategy.stage_states);
        StatesGuard {
            stage_states,
            strategy,
        }
    }
}

impl<'a> Drop for StatesGuard<'a> {
    fn drop(&mut self) {
        std::mem::swap(&mut self.stage_states, &mut self.strategy.stage_states);
    }
}

impl Strategy1F1B {
    fn init_1f1b_devices(world_size: usize) -> Vec<StageState> {
        let mut devices: Vec<StageState> = (0..world_size).map(StageState::new).collect();
        for (rank, state) in devices.iter_mut().enumerate() {
            state.device.prev_stage = if rank == 0 { None } else { Some(rank - 1) };
            state.device.next_stage = if rank == world_size - 1 {
                None
            } else {
                Some(rank + 1)
            };
        }
        devices
    }

    fn fetch_one(&mut self, state: &mut StageState) -> Result<()> {
        let arrangement = self
            .arrangements
            .as_mut()
            .ok_or(anyhow!("Arrangements not initialized"))?;
        let rank = state.device.stage_idx;

        if state.complete(self.num_minibatch) {
            arrangement[rank].push(MiniBatch::Nops);
            state.curr_time += 1;
            return Ok(());
        }

        let next_rank = state.device.next_stage;
        let prev_rank = state.device.prev_stage;

        if rank == 0 {
            if state.curr_time == 0 {
                arrangement[rank].push(MiniBatch::Forward(0));
                state.forward_idx += 1;
                state.curr_time += 1;
                return Ok(());
            }
            let next_rank = next_rank.unwrap();
            let forward_batch = arrangement[rank][state.curr_time - 1].clone();
            let backward_batch = arrangement[next_rank][state.curr_time - 1].clone();

            if let MiniBatch::Backward(idx) = backward_batch {
                debug_assert_eq!(state.backward_idx, idx);
                arrangement[rank].push(MiniBatch::Backward(idx));
                state.backward_idx += 1;
            } else if let MiniBatch::Forward(idx) = forward_batch {
                debug_assert_eq!(state.forward_idx, idx + 1);
                arrangement[rank].push(MiniBatch::Forward(idx + 1));
                state.forward_idx += 1;
            } else {
                return Err(anyhow!(
                    "Invalid state: no forward or backward batch found for rank 0"
                ));
            }
        } else if rank == self.world_size - 1 {
            if state.curr_time == 0 {
                arrangement[rank].push(MiniBatch::Nops);
                state.curr_time += 1;
                return Ok(());
            }
            let prev_rank = prev_rank.unwrap();
            let forward_batch = arrangement[prev_rank][state.curr_time - 1].clone();
            let backward_batch = arrangement[rank][state.curr_time - 1].clone();

            if let MiniBatch::Forward(idx) = backward_batch {
                debug_assert_eq!(state.backward_idx, idx);
                arrangement[rank].push(MiniBatch::Backward(idx));
                state.backward_idx += 1;
            } else if let MiniBatch::Forward(idx) = forward_batch {
                debug_assert_eq!(state.forward_idx, idx);
                arrangement[rank].push(MiniBatch::Backward(idx));
                state.forward_idx += 1;
            } else {
                arrangement[rank].push(MiniBatch::Nops);
            }
        } else {
            if state.curr_time == 0 {
                arrangement[rank].push(MiniBatch::Nops);
                state.curr_time += 1;
                return Ok(());
            }
            let next_rank = next_rank.unwrap();
            let prev_rank = prev_rank.unwrap();
            let forward_batch = arrangement[prev_rank][state.curr_time - 1].clone();
            let backward_batch = arrangement[next_rank][state.curr_time - 1].clone();

            if let MiniBatch::Backward(idx) = backward_batch {
                debug_assert_eq!(state.backward_idx, idx);
                arrangement[rank].push(MiniBatch::Backward(idx));
                state.backward_idx += 1;
            } else if let MiniBatch::Forward(idx) = forward_batch {
                debug_assert_eq!(state.forward_idx, idx);
                arrangement[rank].push(MiniBatch::Forward(idx));
                state.forward_idx += 1;
            } else {
                arrangement[rank].push(MiniBatch::Nops);
            }
        }

        state.curr_time += 1;

        Ok(())
    }
}

impl Strategy for Strategy1F1B {
    fn new(world_size: usize, num_minibatch: usize) -> Self {
        let devices = Self::init_1f1b_devices(world_size);
        Self {
            stage_states: devices,
            world_size,
            num_minibatch,
            arrangements: None,
        }
    }

    fn complete(&self) -> bool {
        self.stage_states
            .iter()
            .all(|d| d.complete(self.num_minibatch))
    }

    fn step(&mut self) -> Result<()> {
        let mut state_guard = StatesGuard::guard(self);
        for state in state_guard.stage_states.iter_mut() {
            state_guard.strategy.fetch_one(state)?;
        }
        Ok(())
    }

    fn arrangements(&mut self) -> Result<Vec<Vec<MiniBatch>>> {
        self.arrangements
            .take()
            .ok_or(anyhow!("Arrangements not initialized"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_guard() {
        const WORLD_SIZE: usize = 4;
        const NUM_MINIBATCH: usize = 8;

        let mut strategy = Strategy1F1B::new(WORLD_SIZE, NUM_MINIBATCH);
        assert!(strategy.stage_states.len() == WORLD_SIZE);
        {
            let guard = StatesGuard::guard(&mut strategy);
            assert!(guard.strategy.stage_states.is_empty());
            assert!(guard.stage_states.len() == WORLD_SIZE);
            for state in guard.stage_states.iter() {
                println!("{state:?}");
            }
        }
        assert!(strategy.stage_states.len() == WORLD_SIZE);
    }
}
