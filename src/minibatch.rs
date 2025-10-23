#[derive(Debug, Clone)]
pub(crate) enum MiniBatch {
    Forward(usize),
    Backward(usize),
    Nops,
}
