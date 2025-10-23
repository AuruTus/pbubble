pub struct Device {
    pub rank: usize,
    pub prev_rank: Option<usize>,
    pub next_rank: Option<usize>,
}

impl Device {
    pub fn new(rank: usize) -> Self {
        Device {
            rank,
            prev_rank: None,
            next_rank: None,
        }
    }
}
