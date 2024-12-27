use linfa::dataset::Records;
use linfa::prelude::Predict;
use linfa::prelude::ToConfusionMatrix;
use linfa::prelude::*;
use linfa::traits::Fit;
use linfa_logistic::LogisticRegression;
use log::info;
use ndarray::s;
use ndarray::{linspace, Array1, Array2};
use plotly::{
    color::{NamedColor, Rgb},
    common::{Marker, Mode},
    layout::{Axis, Layout, TicksDirection},
    Contour, Plot, Scatter,
};

use std::error::Error;
use std::fs::File;
use std::io::Write;

//use ndarray::arr2;
//use ndarray_npy::read_npy;
use ndarray_npy::write_npy;
//use ndarray_npy::ReadNpyExt;
//use ndarray_npy::WriteNpyExt;
//use npy::NpyData;
//use std::env;

//use ndarray::ErrorKind;
use ndarray::ShapeError;
use std::num::ParseFloatError;

use ndarray_npy::ReadNpyError;
use ndarray_npy::WriteNpyError;
//use std::convert::From;

use crate::linear_regression::model;
use crate::linear_regression::predict;
use ndarray::Axis as OtherAxis;
use std::iter::Zip;

#[derive(Debug)]
pub enum Errors {
    ShapeError(ShapeError),
    ParseFloatError(ParseFloatError),
    ReadNpyError(ReadNpyError),
    WriteNpyError(WriteNpyError),
    IoError(std::io::Error),
    MeanCalculationFailed,
}

pub struct OneHiddenLayerNNCache {
    pub Z1: Array2<f32>,
    pub A1: Array2<f32>,
    pub Z2: Array2<f32>,
    pub A2: Array2<f32>,
}

pub struct OneHiddenLayerNNParameters {
    pub W1: Array2<f32>,
    pub b1: Array2<f32>,
    pub W2: Array2<f32>,
    pub b2: Array2<f32>,
}

pub struct GradientDescentResults {
    pub w: Array2<f32>,
    pub b: f32,
    pub dw: Array2<f32>,
    pub db: f32,
    pub costs: Vec<f32>,
}

pub struct ModelResults {
    pub costs: Vec<f32>,
    pub y_prediction_test: Array2<f32>,
    pub y_prediction_train: Array2<f32>,
    pub w: Array2<f32>,
    pub b: f32,
    pub learning_rate: f32,
    pub num_iterations: i32,
}

pub struct PredictionResults {
    pub y_prediction_train: Array2<f32>,
    pub y_prediction_test: Array2<f32>,
}

fn create_array(b: f32) -> Result<Array2<f32>, Errors> {
    /* let result = Array2::from_shape_vec((1, 1), vec![b]).map_err(Errors::ShapeError);
    result */
    Array2::from_shape_vec((1, 1), vec![b]).map_err(Errors::ShapeError)
}

pub fn sigmoid(z: &Array2<f32>) -> Array2<f32> {
    /*
    Compute the sigmoid of z as 1 / (1 + np.exp(-z))
    Apply the exponential function to each element

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z), a probability between 0 and 1
    */

    1.0 / (1.0 + z.mapv(|x| (-x).exp()))
}

pub fn fit_logistic_regression_model(
    train_x: &Array2<f32>,
    train_y: &Array2<f32>,
    test_x: &Array2<f32>,
    test_y: &Array2<f32>,
) -> Result<(ModelResults), Errors> {
    //) -> Result<(ModelResults), Errors> {
    //) -> Result<(), Errors> {
    /*
    This function fits the logistic regression model according to the given training data

    Argument:
    X -- (n_features, n_samples) matrix where features refer to planar coordindates
    Y -- (n_label, n_samples) matrix labels refer to red (0.0) and blue (1.0)

    Returns:
    model
    */

    //let (_train_x, _train_y, _test_x, _test_y) = (x, y, x, y);
    let mut num_iterations = 2000;
    let mut learning_rate = 0.005;
    let print_cost = true;
    let mut costs: Vec<f32> = Vec::new(); // Create an empty vector

    let mut b = 0.0;

    let mut y_prediction_test = create_array(b)?;
    let mut y_prediction_train = create_array(b)?;
    let mut w = create_array(b)?;

    //let (_costs, _y_prediction_test, _y_prediction_train, _w, _b, _learning_rate, _num_iterations) =
    let model_result = model(
        train_x,
        train_y,
        test_x,
        test_y,
        num_iterations,
        learning_rate,
        print_cost,
    );

    // Handle model_results
    match model_result {
        Ok(model_results) => {
            // Process the successful predictions
            (
                costs,
                y_prediction_test,
                y_prediction_train,
                w,
                b,
                learning_rate,
                num_iterations,
            ) = (
                model_results.costs,
                model_results.y_prediction_test,
                model_results.y_prediction_train,
                model_results.w,
                model_results.b,
                model_results.learning_rate,
                model_results.num_iterations,
            );

            let b_array = create_array(b)?;

            // overwrite the file if it exists
            let _ = write_npy("./model/model_weights.npy", &w).map_err(|_| Errors::WriteNpyError);
            let _ =
                write_npy("./model/model_bias.npy", &b_array).map_err(|_| Errors::WriteNpyError);
            let _ = write_npy("./model/y_prediction_train.npy", &y_prediction_train)
                .map_err(|_| Errors::WriteNpyError);
            let _ = write_npy("./model/y_prediction_test.npy", &y_prediction_test)
                .map_err(|_| Errors::WriteNpyError);
        }
        Err(error) => {
            // Handle the error
            eprintln!("Error modeling: {:?}", error);
        }
    }

    Ok(ModelResults {
        costs,
        y_prediction_test,
        y_prediction_train,
        w,
        b,
        learning_rate,
        num_iterations,
    })
}

pub fn linfa_logistic_regression() -> Result<(), Box<dyn Error>> {
    // everything above 6.5 is considered a good wine
    let (train, valid) = linfa_datasets::winequality()
        .map_targets(|x| if *x > 6 { "good" } else { "bad" })
        .split_with_ratio(0.9);

    println!(
        "Fit Logistic Regression classifier with #{} training points",
        train.nsamples()
    );
    info!("train.nsamples: {:?}", train);

    // fit a Logistic regression model with 150 max iterations
    let model = LogisticRegression::default()
        .max_iterations(150)
        .fit(&train)
        .unwrap();

    // predict and map targets
    let pred = model.predict(&valid);

    // create a confusion matrix
    let cm = pred.confusion_matrix(&valid).unwrap();

    // Print the confusion matrix, this will print a table with four entries. On the diagonal are
    // the number of true-positive and true-negative predictions, off the diagonal are
    // false-positive and false-negative
    println!("{:?}", cm);

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    let shape_train = train.records.shape();

    info!("The shape of train is: {:?}", shape_train);

    Ok(())
}

pub fn find_minimum(row: &Array1<f32>) -> f32 {
    let mut x_min = 0.0_f32;
    //row.iter().enumerate().for_each(|(_index, &value)| {
    row.iter().for_each(|&value| {
        if value < x_min {
            x_min = value;
        }
    });
    x_min -= 1.0;

    x_min
}

pub fn find_maximum(row: &Array1<f32>) -> f32 {
    let mut x_max = 0.0_f32;
    // row.iter().enumerate().for_each(|(_index, &value)| {
    row.iter().for_each(|&value| {
        if value > x_max {
            x_max = value;
        }
    });
    x_max += 1.0;

    x_max
}

pub fn meshgrid(x: &Array1<f32>, y: &Array1<f32>) -> Array2<f32> {
    /*
    This function generates 2D grid of points

    Argument:
    x -- single array of grid value along x axis x feature
    y -- single array of grid value along y axis y feature

    Returns:
    2D array (number of features, number of points) ... (2,1000)

    */
    let (nx, _ny) = (x.len(), y.len());
    let mut xx: Array2<f32> = Array2::zeros((2, nx));

    for (i, &x_val) in x.iter().enumerate() {
        xx[[0, i]] = x_val;
    }

    for (i, &y_val) in y.iter().enumerate() {
        xx[[1, i]] = y_val;
    }

    xx
}
