use crate::helper::OneHiddenLayerNNCache;
use crate::helper::OneHiddenLayerNNParameters;
use log::info;
use ndarray::prelude::*;
use rand::thread_rng;
use rand::Rng;
use statrs::distribution::Exp;
use crate::helper::sigmoid;

pub fn tanh(x: &Array2<f32>) -> Array2<f32> {
    (x.mapv(|x| (x).exp()) - (-x).mapv(|x| (x).exp()))
        / (x.mapv(|x| (x).exp()) + (-x).mapv(|x| (x).exp()))
}

pub fn layer_sizes(X: Array2<f32>, Y: Array2<f32>) -> (usize, usize) {
    /*
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- initialise at nn_model()
    n_y -- the size of the output layer
    */

    let n_x = X.shape()[0];
    let n_y = Y.shape()[0];

    (n_x, n_y)
}

fn vec_to_array2(vec: Vec<Vec<f32>>) -> Array2<f32> {
    let rows = vec.len();
    let cols = vec[0].len();

    let mut arr = Array2::zeros((rows, cols));

    for (i, row) in vec.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            arr[[i, j]] = val;
        }
    }

    arr
}

pub fn initialize_parameters(n_x: usize, n_h: usize, n_y: usize) -> OneHiddenLayerNNParameters {
    /*
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    */

    let mut rng = thread_rng(); // creates a thread-local random number generator

    // W1 = np.random.randn(n_h, n_x) * 0.01
    let rows = n_h;
    let cols = n_x;
    let mut random_numbers: Vec<Vec<f32>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        let row: Vec<f32> = (0..cols).map(|_| rng.gen::<f32>() * 0.01).collect();
        random_numbers.push(row);
    }
    let W1: Array2<f32> = vec_to_array2(random_numbers);

    // W2 = np.random.randn(n_y, n_h) * 0.01
    let rows = n_y;
    let cols = n_h;
    let mut random_numbers: Vec<Vec<f32>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        let row: Vec<f32> = (0..cols).map(|_| rng.gen::<f32>() * 0.01).collect();
        random_numbers.push(row);
    }
    let W2: Array2<f32> = vec_to_array2(random_numbers);

    // b1 = np.zeros((n_h, 1))
    // b2 = np.zeros((n_y, 1))
    let b1: Array2<f32> = Array2::zeros((n_h, 1));
    let b2: Array2<f32> = Array2::zeros((n_y, 1));

    // parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    let W1_owned = W1.to_owned();
    let W2_owned = W2.to_owned();
    let b1_owned = b1.to_owned();
    let b2_owned = b2.to_owned();
    OneHiddenLayerNNParameters {
        W1: W1_owned,
        b1: b1_owned,
        W2: W2_owned,
        b2: b2_owned,
    }
}

pub fn compute_cost(A2: &Array2<f32>, Y: &Array2<f32>) -> f32 {
    /*
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost given equation (13)

    */

    let m: f32 = Y.shape()[1] as f32; // number of examples

    // Compute the cross-entropy cost

    /*
    let logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1.0 - A2), 1.0 - Y);
    let cost = -np.sum(logprobs) / m;
    */
    let a = A2;
    let y = Y;
    let cost: f32 =
        -(y * (a.mapv(|e| e.ln())) + (1.0 - y) * ((1.0 - a).mapv(|d| d.ln()))).sum() / m;

    cost
}
pub fn forward_propagation(
    X: &Array2<f32>,
    parameters: &OneHiddenLayerNNParameters,
) -> (Array2<f32>, OneHiddenLayerNNCache) {
    /*
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    */

    //Retrieve each parameter from the dictionary "parameters"
    let W1 = parameters.W1.clone(); // (n_h, n_x)
    let b1 = parameters.b1.clone(); // (n_h, 1)
    let W2 = parameters.W2.clone(); // (1, n_h)
    let b2 = parameters.b2.clone(); // (1, 1)

    // Implement Forward Propagation to calculate A2 (probabilities)

    let Z1 = W1.dot(X) + b1; // (n_h, n_x) . (n_x, m) -> (n_h, m)
    let A1 = tanh(&Z1); // (n_h, m)

    let Z2 = W2.dot(&A1) + b2;   // (1, n_h) . (n_h, m) -> (1,m)
    let A2 = sigmoid(&Z2); // (1,m)

    assert_eq!(A2.shape(), &[1, X.shape()[1]]);  

    let z1_owned = Z1.to_owned();
    let a1_owned = A1.to_owned();
    let z2_owned = Z2.to_owned();
    let a2_owned = A2.to_owned();

    let cache = OneHiddenLayerNNCache {
        Z1: z1_owned,
        A1: a1_owned,
        Z2: z2_owned,
        A2: a2_owned,
    };

    (A2, cache)
}
pub fn nn_model(
    X: &Array2<f32>,
    Y: &Array2<f32>,
    n_h: usize,
    num_iterations: i32,
    print_cost: bool,
) -> (OneHiddenLayerNNParameters, Vec<f32>) {
    /*
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    */

    let (n_x, n_y) = layer_sizes(X.clone(), Y.clone());
    let mut costs: Vec<f32> = Vec::new();
    let parameters = initialize_parameters(n_x, n_h, n_y);

    for i in 0..num_iterations {
        // Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".

        let (A2, cache) = forward_propagation(X, &parameters);
        let cost: f32 = compute_cost(&A2, &Y);
        //grads = backward_propagation(parameters, cache, X, Y)
        // parameters = update_parameters(parameters, grads)

        // Print the cost every 1000 iterations
        let print_interval = 1000;
        if i % print_interval == 0 {
            costs.push(cost);

            if print_cost {
                println!("Cost after iteration {:?}: {:?}", i, cost);
                info!("Cost after iteration {:?}: {:?}", i, cost);
            }
        }
    }
    (parameters, costs)
}
