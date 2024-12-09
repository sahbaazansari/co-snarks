use noir_mpc_ml::ml::train_multi_class;
use noir_mpc_ml::quantized::Quantized;

fn main(
    inputs: [[Quantized; 4]; 30],
    labels: [[Quantized; 30]; 3],
) -> pub [([Quantized; 4], Quantized); 3] {
    let learning_rate = 6554;
    let ratio = 2185;

    let epochs = 2;

    let learning_rate = Quantized::new(learning_rate);
    let ratio = Quantized::new(ratio);
    let parameters = train_multi_class(epochs, inputs, labels, learning_rate, ratio);
    parameters
}
