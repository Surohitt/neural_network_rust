mod xor_input;
mod xor_model;
fn main() {
    let layer = xor_model::input_layer();
    let (x_in, y_out) = xor_input::get_xor_data();
    println!("Our numbers are: {:?}", layer);
    println!("X input is : {:?}", x_in);
    println!("y output: {:?}", y_out);
    let perceptron = xor_model::Perceptron {
        bias: 1.0,
        lr: 0.05,
        input_layer: layer,
        node_values: layer,
    };
    println!("Perceptron: {:?}", perceptron);
    let mut trained_perceptron: xor_model::Perceptron = train(perceptron, (x_in, y_out));
    println!("Trained Perceptron: {:?}", trained_perceptron);
    for i in 0..4 {
        let output = trained_perceptron.forward(x_in[i]);
        println!("Output: {}", output);
        println!("Expected: {}", y_out[i]);
        println!("layers: {:?}", trained_perceptron.input_layer);
    }
}

fn train(
    mut perceptron: xor_model::Perceptron,
    data: ([[f64; 2]; 4], [i8; 4]),
) -> xor_model::Perceptron {
    let num_iters = 1000;
    let input_data = data.0;
    let output_data = data.1;
    let mut current_iteration = 0;

    let mut num_correct: Vec<Vec<i32>> = vec![vec![0; 4]; num_iters];

    while current_iteration < num_iters {
        for row in 0..output_data.len() {
            let classification = perceptron.classify(input_data[row]);
            if classification == output_data[row] {
                num_correct[current_iteration][row] = 1;
            } else {
                let output = perceptron.forward(input_data[row]);
                perceptron.update_weights(output_data[row], output)
            }
        }
        if current_iteration % 200 == 0 {
            println!("Weights: {:?}", perceptron.input_layer);
            println!("Bias: {:?}", perceptron.bias);
        }
        current_iteration += 1;
    }
    let mut _total_correct = 0;
    for i in 0..num_correct.len() {
        for ind in 0..4 {
            if num_correct[i][ind] == 1 {
                _total_correct += 1
            }
        }
    }
    perceptron
}
