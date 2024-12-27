// use fern::Dispatch;
use log::info;
use log::LevelFilter;
use ndarray::Array2;
use one_hidden_layer_nn::data::injest;
use one_hidden_layer_nn::helper::fit_logistic_regression_model;
use one_hidden_layer_nn::helper::ModelResults;
use one_hidden_layer_nn::onehiddenlayer_nn::nn_model;
use one_hidden_layer_nn::plot::plot;
use one_hidden_layer_nn::plot::plot_decision_boundary;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the logger
    let _ = fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{} [{}] [{}] {}",
                chrono::Local::now().format("%d-%m-%Y %H:%M:%S"),
                record.level(),
                record.target(),
                message
            ))
        })
        .level(LevelFilter::Debug)
        .chain(fern::log_file("./log/info.log")?)
        .apply();

    // Prepare datasets
    let a = 10; // maximum ray of the flower, length of petal
    let m_train = 180; // number of examples or points of train dataset
    let train_test_split_ratio = 3;
    let m_test = m_train / train_test_split_ratio; // number of examples or points of test dataset

    let mut dataset_title = "train_dataset";
    let (train_x, train_y) = injest(m_train, a, dataset_title);

    dataset_title = "test_dataset";
    let (test_x, test_y) = injest(m_test, a, dataset_title);

    // Visualise datasets
    let plot_title = "train_dataset";
    let plot_main = plot(&train_x, &train_y, a, plot_title);

    let plot_title = "test_dataset";
    let _ = plot(&test_x, &test_y, a, plot_title);

    // linear regression model

    // let _ = linfa_logistic_regression(); // testing of external linfa crate to be continued

    let mut model_lr = ModelResults {
        costs: Vec::new(),
        y_prediction_test: Array2::zeros((1, m_test)),
        y_prediction_train: Array2::zeros((1, m_train)),
        w: Array2::zeros((test_x.shape()[0], 1)),
        b: 0.0,
        learning_rate: 0.0,
        num_iterations: 0,
    };

    let model_result = fit_logistic_regression_model(&train_x, &train_y, &test_x, &test_y);

    match model_result {
        Ok(model_results) => {
            // Process the successful predictions
            model_lr = model_results;
        }
        Err(error) => {
            // Handle the error
            eprintln!("Error modeling: {:?}", error);
        }
    }

    // visualise the decision boundary plot
    let plot_title = "linear regression decision boundary";
    plot_decision_boundary(&train_x, model_lr, plot_title, plot_main);

    // one hidden layer NN with n_h-dimensional hidden layer
    let n_h: usize = 4;
    let num_iterations: i32 = 10000;
    let print_cost: bool = false;
    let (parameters, costs) = nn_model(&train_x, &train_y, n_h, num_iterations, print_cost);

    Ok(())
}
