import torch
import numpy as np
from data import yahooFinance

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from typing import Tuple
from models import Model
from data import Dataset
from utils import train, inference, runConfig, config, read_file, set_seed


# Train shadow models where each one is trained 
def train_shadow_models(config: runConfig, all_symbols: list[str], num_shadow_models: int=3) -> list[Tuple[Model, Dataset]]:
    shadow_models = []
    np.random.shuffle(all_symbols)

    for i in range(num_shadow_models):
        # Create a dataset for the shadow model by sampling the all_symbols superset
        subset = all_symbols[i::num_shadow_models]  
        print(f"Shadow Model {i + 1} training on: {subset}")

        # Get the data for this sahdow model
        config.symbols = subset
        data = yahooFinance(config)
        trainX, trainY, _, _ = data.getData()

        # Train shadow model
        shadow_model = config.architecture(
            config.hidden_dim, config.layers, config.window_size, 1
        )
        shadow_model = train(shadow_model, trainX, trainY, config)
        shadow_models.append((shadow_model, data))

    return shadow_models


# Create a dataset for the attack model using the output of each shadow model
def prepare_attack_data(shadow_models: list[Tuple[Model, Dataset]]) -> Tuple[np.ndarray, np.ndarray]:
    attack_features = []
    attack_labels = []
    for model, data in shadow_models:
        trainX, trainY, testX, testY = data.getData()
        # Get predictions from the shadow model
        train_preds = inference(model, testX, data, v=0)
        test_preds = inference(model, testX, data, v=0)

        # Create a dataset for training an attack model
        train_confidences = np.abs(train_preds - testY)
        test_confidences = np.abs(test_preds - testY)
        attack_features.extend(train_confidences.tolist())
        attack_features.extend(test_confidences.tolist())
        attack_labels.extend([1] * len(train_confidences))
        attack_labels.extend([0] * len(test_confidences))

    return np.array(attack_features), np.array(attack_labels)

# Train an attack model (classifier that predicts member or non-member)
def train_attack_model(attack_features: np.ndarray, attack_labels: np.ndarray) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        attack_features, attack_labels, test_size=0.3, random_state=42
    )
    attack_model = RandomForestClassifier() # Can be any binary classifier
    attack_model.fit(X_train.reshape(-1, 1), y_train)
    
    # Evaluate attack model on the created dataset
    predictions = attack_model.predict(X_test.reshape(-1, 1))
    accuracy = accuracy_score(y_test, predictions)
    print(f"Attack Model Accuracy: {accuracy:.4f}")
    return attack_model



# Attack the target model
def attack_target_model(target_model: Model, all_symbols: list[str], attack_model: RandomForestClassifier, config: runConfig):
    results = {}
    for symbol in all_symbols:
        print(f"Querying target model with stock: {symbol}")

        config.symbols = [symbol]
        data = yahooFinance(config)
        _, _, testX, testY = data.getData()

        # Query the target model
        target_preds = inference(target_model, testX, data, v=0)
        confidences = np.abs(target_preds - testY).mean()

        # Use the attack model to infer membership
        membership = attack_model.predict([[confidences]])
        results[symbol] = "MEMBER" if membership == 1 else "NON-MEMBER"
        print(f"Stock: {symbol}, Membership: {results[symbol]}")

    return results







# Actually perform the attack using shadow models
def perform_shadow_attack(config: runConfig, weights_path: str="example_model.pth") -> None:
    # Save the answers for later reference and load the list of stocks to investigate (superset)
    answers = config.symbols
    all_symbols = read_file("all_symbols.txt")

    # Create shadow models (3 here)
    shadow_models = train_shadow_models(run_config, all_symbols)
    
    # Prepare data and attack model
    attack_features, attack_labels = prepare_attack_data(shadow_models)
    attack_model = train_attack_model(attack_features, attack_labels)
    
    # Load the target model
    model = config.architecture(
        config.hidden_dim, config.layers, config.window_size, 1
    )
    model.load_state_dict(torch.load(weights_path, weights_only=True))

    # For each possible symbol make a prediction on if it's in the training data
    results = {}
    for symbol in all_symbols:
        print(f"Querying target model with stock: {symbol}")

        # Get a dataset of just the test (target) symbol
        config.symbols = [symbol]
        data = yahooFinance(config)
        _, _, testX, testY = data.getData()

        # Query the target model
        target_preds = inference(model, testX, data, v=0)
        confidences = np.abs(target_preds - testY).mean()

        # Predict membership with teh attack model
        membership = attack_model.predict([[confidences]])
        results[symbol] =  membership == 1 
    
    evaluate_attack(answers, results, v=1)


if __name__ == "__main__":
    run_config = config()
    perform_shadow_attack(run_config)

    print("\n\t-fin-\n")
