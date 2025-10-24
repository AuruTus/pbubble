/// [`Stage`] processing stage in the pipeline.
///
/// In interleaved pipeline parallelism strategy, each device holds multiple
/// stages, which is also called `virtual stage`.
///
#[derive(Debug, Clone)]
pub struct Stage {
    pub stage_idx: usize,
    pub prev_stage: Option<usize>,
    pub next_stage: Option<usize>,
}

impl Stage {
    pub fn new(stage_idx: usize) -> Self {
        Stage {
            stage_idx,
            prev_stage: None,
            next_stage: None,
        }
    }
}
