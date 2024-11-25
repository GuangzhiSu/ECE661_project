from data import yahooFinance
from utils import (
    parse_args,
    set_seed,
    train,
    inference,
    evaluation_metric,
    make_a_plot,
    save_model,
    runConfig,
    read_file,
)
from models import MambaStock, Transformer, LSTM


# Hard coded config for an example run
def config():
    # Data
    file_path = "symbols.txt"
    symbols = read_file(file_path)
    start_date = "2010-01-01"
    train_split = 0.8

    # Model
    architecture = LSTM
    # architecture = MambaStock
    window_size = 20
    hidden_dim = 16
    layers = 2

    # Training
    epochs = 100
    learning_rate = 0.01
    weight_decay = 1e-5
    cuda = False

    return runConfig(
        symbols,
        start_date,
        train_split,
        architecture,
        window_size,
        hidden_dim,
        layers,
        epochs,
        learning_rate,
        weight_decay,
        cuda,
    )


# Hard coded config for a single stock to graph predcitions of
def graph_config(config):
    # Data
    file_path = "graph.txt"
    symbols = read_file(file_path)
    start_date = "2010-01-01"
    train_split = 0.5

    return runConfig(
        symbols,
        start_date,
        train_split,
        config.architecture,
        config.window_size,
        config.hidden_dim,
        config.layers,
        config.epochs,
        config.learning_rate,
        config.weight_decay,
        config.cuda,
    )


def create_model():
    run_config = config()  # Hardcoded values from above
    # run_config = parse_args() # Parse cli args

    # Data
    data = yahooFinance(run_config)
    trainX, trainY, testX, testY = data.getData()

    # Setup and train model
    set_seed(1234)
    model = run_config.architecture(
        run_config.hidden_dim, run_config.layers, run_config.window_size, 1
    )
    model = train(model, trainX, trainY, run_config)

    # Run inference and calculate error
    model_preds = inference(model, testX, data)
    scores = evaluation_metric(testY, model_preds)

    # Save the model
    save_model(model, "example_model.pth")

    # Save a graph of predictions on a new stock
    data = yahooFinance(graph_config(run_config))
    trainX, trainY, testX, testY = data.getData()
    # model = train(model, trainX, trainY, run_config)  # This line finetunes the model on the 'train' portion of the new stock
    model_preds = inference(model, testX, data)
    scores = evaluation_metric(testY, model_preds)
    file_path = f"test.png"
    make_a_plot(data, model_preds, file_path, "Graph Title")


# Do that thing
if __name__ == "__main__":
    create_model()
    print("\n\t-fin-\n")
