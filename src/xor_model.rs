use assert_approx_eq::assert_approx_eq;
use rand::Rng;

#[derive(Debug)]
pub struct Perceptron {
    pub bias: f64,
    pub lr: f64,
    pub input_layer: [f64; 2],
    pub node_values: [f64; 2],
}

impl Perceptron {
    pub fn _gradient(&self, expected: f64, output: f64, node: f64) -> f64 {
        node * (expected - output)
    }
    pub fn update_weights(&mut self, expected: i8, output: f64) {
        let mut expected_float: f64 = 0.0;
        if expected == 1 {
            expected_float = 1.0;
        } else {
            expected_float = 0.0;
        }
        for ind in 0..self.input_layer.len() {
            self.input_layer[ind] +=
                self.lr * self._gradient(expected_float, output, self.node_values[ind])
        }
        self.bias += self.lr * self._gradient(expected_float, output, self.bias);
    }
    pub fn forward(&mut self, input_data: [f64; 2]) -> f64 {
        let output_value = self.bias + self.dot(self.input_layer, input_data);
        output_value
    }
    pub fn classify(&mut self, input_data: [f64; 2]) -> i8 {
        let output = self.forward(input_data);
        let threshold = 0.5;
        let return_val: i8;
        if output >= threshold {
            return_val = 1;
        } else {
            return_val = 0;
        }
        return_val
    }
    pub fn dot(&mut self, a: [f64; 2], b: [f64; 2]) -> f64 {
        let mut output: f64 = 0.0;

        for ind in 0..a.len() {
            output += a[ind] * b[ind];
            self.node_values[ind] = a[ind] * b[ind];
        }
        output
    }
}

pub fn input_layer() -> [f64; 2] {
    let mut rng = rand::thread_rng();
    let x_1 = rng.gen_range(0.0, 1.0);
    let x_2 = rng.gen_range(0.0, 1.0);
    [x_1, x_2]
}

#[test]
fn test_dot_product_method() {
    let layer = input_layer();
    let mut perceptron = Perceptron {
        bias: 1.0,
        lr: 0.05,
        input_layer: layer,
        node_values: layer,
    };
    let output = perceptron.dot([0.51, 0.52], [0.31, 0.32]);
    let expected_output = 0.3245;
    println!(
        "Output: {:?}, Expected Output: {:?}",
        output, expected_output
    );
    assert_approx_eq!(output, expected_output);
}

#[test]
fn test_forward_pass() {
    let layer = [0.5, 0.5];
    let mut perceptron = Perceptron {
        bias: 1.0,
        lr: 0.05,
        input_layer: layer,
        node_values: layer,
    };
    let output = perceptron.forward([0.0, 1.0]);
    println!("Output: {:?}, Expected Output: {:?}", output, 1.5);
    assert_eq!(output, 1.5);
}

#[test]
fn test_classify() {
    let layer = [0.5, 0.5];
    let mut perceptron = Perceptron {
        bias: 1.0,
        lr: 0.05,
        input_layer: layer,
        node_values: layer,
    };
    let classification = perceptron.classify([0.0, 1.0]);
    println!("Output: {:?}, Expected Output: {:?}", classification, 1);
    assert_eq!(classification, 1);
}

#[test]
fn test_update_weights() {
    let layer = [0.1, 0.1];
    let mut perceptron = Perceptron {
        bias: 1.0,
        lr: 0.05,
        input_layer: layer,
        node_values: layer,
    };
    let input = [1.0, 1.0];
    let expected = 0;
    let output = 1.2;

    perceptron.update_weights(expected, output);

    let layer2 = perceptron.input_layer;
    let bias2 = perceptron.bias;

    assert_ne!(layer2, layer);
    assert_ne!(bias2, 1.0);
    assert_eq!(layer2, [0.094, 0.094]);
    assert_eq!(bias2, 0.94);

    perceptron.update_weights(expected, output);

    let layer3 = perceptron.input_layer;
    let bias3 = perceptron.bias;

    assert_ne!(layer3, layer);
    assert_ne!(bias3, 1.0);
    assert_eq!(layer3, [0.088, 0.088]);
    assert_approx_eq!(bias3, 0.884, 1e-3f64);
}
