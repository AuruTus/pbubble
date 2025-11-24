#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MiniBatch {
    Forward(usize),
    Backward(usize),
    Nops,
}
