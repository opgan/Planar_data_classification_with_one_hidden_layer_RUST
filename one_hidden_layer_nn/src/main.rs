// use fern::Dispatch;
use log::info;
use log::LevelFilter;
use ndarray::Array2;
use one_hidden_layer_nn::data::injest;
use one_hidden_layer_nn::helper::fit_logistic_regression_model;
use one_hidden_layer_nn::helper::ModelResults;
use one_hidden_layer_nn::plot::plot;
use one_hidden_layer_nn::plot::plot_decision_boundary;
use one_hidden_layer_nn::plot::simple_contour_plot;

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

    let (train_x, train_y) = injest(m_train, a);

    let shape_x = train_x.shape();
    let shape_y = train_y.shape();
    let m_examples = shape_y[1];

    println!("The shape of x is: {:?}", shape_x);
    println!("The shape of y is: {:?}", shape_y);
    println!("There are m = {:?} training examples ", m_examples);

    info!("The shape of x is: {:?}", shape_x);
    info!("The shape of y is: {:?}", shape_y);
    info!("There are m = {:?} training examples ", m_examples);

    let (test_x, test_y) = injest(m_test, a);

    let shape_x = test_x.shape();
    let shape_y = test_y.shape();
    let m_examples = shape_y[1];

    println!("The shape of x is: {:?}", shape_x);
    println!("The shape of y is: {:?}", shape_y);
    println!("There are m = {:?} testing examples ", m_examples);

    info!("The shape of x is: {:?}", shape_x);
    info!("The shape of y is: {:?}", shape_y);
    info!("There are m = {:?} testing examples ", m_examples);

    // Visualise datasets
    let plot_title = "train dataset";
    let mut plot_main = plot(&train_x, &train_y, a, plot_title);

    let plot_title = "test dataset";
    let _ = plot(&test_x, &test_y, a, plot_title);

    //let _ = linfa_logistic_regression();

    let mut model_lr = ModelResults {
        costs: Vec::new(),
        y_prediction_test: Array2::zeros((1, m_test)),
        y_prediction_train: Array2::zeros((1, m_train)),
        w: Array2::zeros((shape_x[0], 1)),
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

    //info!("main train_x shape is: {:?} ", train_x.shape());
    //info!("main model.w shape is: {:?} ", model_lr.w.shape());
    //info!("main model.b is: {:?} ", model_lr.b);

    let plot_title = "decision boundary";
    plot_decision_boundary(&train_x, model_lr, plot_title, plot_main);

    //compute_accuracy(modelLR, train_y, test_y);

    //simple_contour_plot(plot_title); // trying out example contour plot codes from plotly

    Ok(())
}
