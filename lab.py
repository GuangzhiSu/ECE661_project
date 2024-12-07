from data import yahooFinance
from utils import (
    parse_args,
    set_seed,
    train,
    inference,
    evaluation_metric,
    make_a_plot,
    save_model,
    config,
    graph_config,
)
from attacks import perform_attack, perform_shadow_attack
import os
import torch


def main():
    # Get/Create a run configuration
    run_config = config()  # Hardcoded values from utils file
    # run_config = parse_args() # Parse cli args

    # Check if the trained model file exists
    if os.path.exists("DP.pth"):
        # If model exists, load it and proceed directly to the attack part
        set_seed(1234)
        model = run_config.architecture(
            run_config.hidden_dim, run_config.layers, run_config.window_size, 1
        )
        model.load_state_dict(torch.load("DP.pth"))
        
    else:
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
        save_model(model, "DP.pth")

        # Save a graph of predictions on a new stock
        data = yahooFinance(graph_config(run_config))
        trainX, trainY, testX, testY = data.getData()
        # model = train(model, trainX, trainY, run_config)  # This line finetunes the model on the 'train' portion of the new stock
        model_preds = inference(model, testX, data)
        scores = evaluation_metric(testY, model_preds)
        make_a_plot(data, model_preds, f"test.png", "Graph Title")

    # Run a membership inference attack on the trained model
    # perform_attack(run_config)

    # Run a more sophisticated MIA with shadow models
    # perform_shadow_attack(run_config)


# Do that thing
if __name__ == "__main__":
    main()
    print("\n\t-fin-\n")
