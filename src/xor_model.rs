use rand::Rng;

#[derive(Debug)]
pub struct Perceptron{
    pub bias: f64,
    pub lr: f64,
    pub input_layer: [f64;2],
    pub node_values: [f64;2],
}

impl Perceptron {
    pub fn _gradient(&self,expected: f64, output: f64, node: f64)->f64{
        node * (expected-output)
    }
    pub fn update_weights(&mut self, expected:i8, output:f64){
        let mut expected_float: f64 = 0.0;
        if expected==1{
            expected_float = 1.0;
        } else {
            expected_float = 0.0;
        }
        for ind in 0..self.input_layer.len() {
            self.input_layer[ind] += self.lr * self._gradient(expected_float, output, self.node_values[ind])
        }
        self.bias += self.lr * self._gradient(expected_float, output, 1.0);
    }
    pub fn forward(&mut self, input_data:[f64;2]) -> f64{
        let output_value = self.bias + self.dot(self.input_layer, input_data);
        output_value
    }
    pub fn classify(&mut self, input_data:[f64;2])->i8{
        let output = self.forward(input_data);
        let threshold = 0.5;
        let return_val: i8;
        if output>=threshold{
            return_val = 1;
        } else {
            return_val = 0;
        }
        return_val
    }
    pub fn dot(& mut self,a:[f64;2], b:[f64;2])-> f64{
        let mut output:f64 = 0.0;

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
