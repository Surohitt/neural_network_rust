pub fn get_xor_data() -> ([[f64; 2]; 4], [i8; 4]) {
    let mut x_in = [[0f64; 2]; 4];
    // let y_out : [i8; 4] = [0, 1 ,1, 0];
    let y_out: [i8; 4] = [1, 1, 1, 0];

    for (ind, row) in x_in.iter_mut().enumerate() {
        if ind == 0 {
            row[0] = 0.0;
            row[1] = 0.0;
        } else if ind == 1 {
            row[0] = 0.0;
            row[1] = 1.0;
        } else if ind == 2 {
            row[0] = 1.0;
            row[1] = 0.0;
        } else {
            row[0] = 1.0;
            row[1] = 1.0;
        }
    }
    (x_in, y_out)
}

#[test]
fn assert_xor_data_assembled() {
    let data = get_xor_data();
    println!("Data: {:?}", data);
    assert_eq!(data.1.len(), 4);
}
