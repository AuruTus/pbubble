use crate::minibatch::MiniBatch;

pub fn print_arrangements_matrix(arrangements: &[Vec<MiniBatch>]) {
    if arrangements.is_empty() {
        return;
    }

    let max_steps = arrangements.iter().map(|arr| arr.len()).max().unwrap_or(0);

    // Print header
    print!("Step\t");
    for step in 0..max_steps {
        print!("{:4} ", step);
    }
    println!();

    // Print each rank's schedule
    for (rank, arr) in arrangements.iter().enumerate() {
        print!("Rank {}\t", rank);
        for step in 0..max_steps {
            if step < arr.len() {
                match &arr[step] {
                    MiniBatch::Forward(idx) => print!(" F{:02} ", idx),
                    MiniBatch::Backward(idx) => print!(" B{:02} ", idx),
                    MiniBatch::Nops => print!(" --- "),
                }
            } else {
                print!(" --- ");
            }
        }
        println!();
    }
}
