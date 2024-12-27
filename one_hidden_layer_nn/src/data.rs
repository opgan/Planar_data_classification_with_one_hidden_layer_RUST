// use crate::helper::generate_flower_planar_dataset;
use log::info;
//use ndarray::Array2;
use ndarray::{linspace, Array1, Array2};
use rand::thread_rng;
use rand::Rng;

pub fn injest(m: usize, a: i32, title: &str) -> (Array2<f32>, Array2<f32>) {
    /*
        Load dataset

        Arguments:
        m -- number of examples
        a -- length of flower petal
        title -- name of dataset

        Return:
        X -- (features, examples) array
        Y -- (1, examples) array
    */

    let (x, y) = generate_flower_planar_dataset(m, a); // Loading data

    let shape_x = x.shape();
    let shape_y = y.shape();
    let m_examples = shape_y[1];

    println!("The shape of x is: {:?} in {:?}", shape_x, title);
    println!("The shape of y is: {:?} in {:?}", shape_y, title);
    println!("There are m = {:?} examples in {:?}", m_examples, title);

    info!("The shape of x is: {:?} in {:?}", shape_x, title);
    info!("The shape of y is: {:?} in {:?}", shape_y, title);
    info!("There are m = {:?} examples in {:?}", m_examples, title);

    (x, y)
}

pub fn generate_spiral_planar_dataset() -> (Array2<f32>, Array2<f32>) {
    /*
        Generate spiral-shaped  data

        Arguments:
        none

        Return:
        X -- (2, 400) array
        Y -- (1, 400) array
    */

    // generate spiral-shaped data points
    const M: usize = 400; // number of examples or points
    const N: usize = M / 2; // number of points per class
    let d = 2; // number of columns in X
    let mut x: Array2<f32> = Array2::zeros((M, d)); // m rows of points and D columns of coordinates
    let mut y: Array2<f32> = Array2::zeros((M, 1)); // labels vector (0.0 for red, 1.0 for blue)
    let a = 4; // # maximum ray of the flower, length of petal

    for j in 0..2 {
        // generate data for 2 classes

        // r is the radius, and t is the angle in radians

        // np.random.randn(N)*0.2
        let mut rng = thread_rng(); // creates a thread-local random number generator
        let random_numbers: Vec<f32> = (0..N) // N random numbers collected into a vector Vec<f64>
            .map(|_| rng.gen::<f32>() * 0.2) // scale the random number by multiplying it with 0.2
            .collect();
        // info!("random_numbers for t is: {:?}", random_numbers);

        // t = np.linspace(j*3.12,(j+1)*3.12,N)
        let start = j as f32 * 3.12;
        let end = (j as f32 + 1.0) * 3.12;
        let t = linspace(start, end, N);
        let mut t: Vec<f32> = t.collect();

        // iterate t vector and add random values
        // t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2

        random_numbers
            .iter()
            .enumerate()
            .for_each(|(index, value)| {
                t[index] += *value; // add random value to t for theta angle
            });

        // r = a*np.sin(4*t)
        let mut r: Vec<f32> = t.clone();
        // let mut rr: f32=0.0;

        let mut modified_r = Vec::with_capacity(r.len());
        //r.iter().enumerate().for_each(|(_index, value)| {
        r.iter().for_each(|value| {
            let rr = *value * 4.0;
            modified_r.push((a as f32) * rr.sin());
        });
        r = modified_r;
        // info!("r is: {:?}", r);

        // iterate r vector and add random values
        // r = a*np.sin(4*t) + np.random.randn(N)*0.2

        let random_numbers: Vec<f32> = (0..N) // N random numbers collected into a vector Vec<f64>
            .map(|_| rng.gen::<f32>() * 0.2) // scale the random number by multiplying it with 0.2
            .collect();

        //info!("random_numbers for r is: {:?}", random_numbers);

        random_numbers
            .iter()
            .enumerate()
            .for_each(|(index, value)| {
                r[index] += *value;
            });
        // info!("r is: {:?}", r);

        // stacking column-wise vector of coordinates (r*np.sin(t), r*np.cos(t)) vertically into a single array, where each row rep a point
        let mut tx: f32 = 0.0;
        let mut rx: f32 = 0.0;

        for ix in N * j..N * (j + 1) {
            if ix < N {
                tx = t[ix];
                rx = r[ix];
            } else {
                tx = t[ix - N];
                rx = r[ix - N];
            }
            x[[ix, 0]] = rx * (tx.sin());
            x[[ix, 1]] = rx * (tx.cos());
            y[[ix, 0]] = j as f32;
        }
    }

    let xt = x.t().to_owned();
    let yt = y.t().to_owned();

    //info!("xt is: {:?}", xt);
    //info!("yt is: {:?}", yt);
    (xt, yt)
}

pub fn generate_flower_planar_dataset(m: usize, a: i32) -> (Array2<f32>, Array2<f32>) {
    /*
        Generate spiral-shaped  data

        Arguments:
        m: number of points or examples
        a:amplitude or length of petal

        Return:
        X -- (2, 400) array
        Y -- (1, 400) array
    */

    // generate spiral-shaped data points
    // const M: usize = 400; // number of examples or points
    // const N: usize = M / 2; // number of points per class
    let n = m / 2; // 2 classes of equal number of examples
    let d = 2; // assuming 2 = number of columns in X
    let mut x: Array2<f32> = Array2::zeros((m, d)); // m rows of points and D columns of coordinates
    let mut y: Array2<f32> = Array2::zeros((m, 1)); // labels vector (0.0 for red, 1.0 for blue)
                                                    // let a = 4; // # maximum ray of the flower, length of petal

    for j in 0..2 {
        // generate data for 2 classes j is red class 0  or blue class 1

        // r is the radius, and t is the angle in radians

        // np.random.randn(N)*0.2
        let mut rng = thread_rng(); // creates a thread-local random number generator
        let random_numbers: Vec<f32> = (0..n) // N random numbers collected into a vector Vec<f64>
            .map(|_| rng.gen::<f32>() * 0.2) // scale the random number by multiplying it with 0.2
            .collect();
        // info!("random_numbers for t is: {:?}", random_numbers);

        // t = np.linspace(j*3.12,(j+1)*3.12,N)
        let start = j as f32 * 3.12;
        let end = (j as f32 + 1.0) * 3.12;
        let t = linspace(start, end, n);
        let mut t: Vec<f32> = t.collect();

        // iterate t vector and add random values
        // t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2

        random_numbers
            .iter()
            .enumerate()
            .for_each(|(index, value)| {
                t[index] += *value; // add random value to t for theta angle
            });

        // r = a*np.sin(4*t)
        let mut r: Vec<f32> = t.clone();

        let mut modified_r = Vec::with_capacity(r.len());
        //r.iter().enumerate().for_each(|(_index, value)| {
        r.iter().for_each(|value| {
            let rr = *value * 4.0;
            modified_r.push((a as f32) * rr.sin());
        });
        r = modified_r;

        // iterate r vector and add random values
        // r = a*np.sin(4*t) + np.random.randn(N)*0.2

        let random_numbers: Vec<f32> = (0..n) // N random numbers collected into a vector Vec<f64>
            .map(|_| rng.gen::<f32>() * 0.2) // scale the random number by multiplying it with 0.2
            .collect();

        random_numbers
            .iter()
            .enumerate()
            .for_each(|(index, value)| {
                r[index] += *value;
            });

        // stacking column-wise vector of coordinates (r*np.sin(t), r*np.cos(t)) vertically into a single array, where each row rep a point
        let mut tx: f32 = 0.0;
        let mut rx: f32 = 0.0;

        for ix in n * j..n * (j + 1) {
            if ix < n {
                tx = t[ix];
                rx = r[ix];
            } else {
                tx = t[ix - n];
                rx = r[ix - n];
            }
            x[[ix, 0]] = rx * (tx.sin());
            x[[ix, 1]] = rx * (tx.cos());
            y[[ix, 0]] = j as f32;
        }
    }

    let xt = x.t().to_owned();
    let yt = y.t().to_owned();

    (xt, yt)
}
