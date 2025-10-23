#[derive(Debug, Clone)]
pub enum MiniBatch {
    Forward(usize),
    Backward(usize),
    Nops,
}
