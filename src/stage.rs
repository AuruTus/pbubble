pub struct Stage {
    pub rank: usize,
    pub prev_rank: Option<usize>,
    pub next_rank: Option<usize>,
}

impl Stage {
    pub fn new(rank: usize) -> Self {
        Stage {
            rank,
            prev_rank: None,
            next_rank: None,
        }
    }
}
