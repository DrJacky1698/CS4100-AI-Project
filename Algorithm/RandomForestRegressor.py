import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

import joblib


# this is a techinique to extract information from FEN called "feature hashing"
# " Feature hashing involves converting features into a fixed size and format that can be processed by machine 
# learning algorithms. This is particularly useful for handling high-dimensional data with many unique features. "
# and there is indeed some other approach such as "vectorization", which tokenizes strings and counting 
# the occurrences of these tokens in each document. But we are not using it.
# def extract_features_from_fen(fen):
    # split the FEN string into its components
    # parts = fen.split(' ')
    # piece_placement = parts[0]

    # # initialize a dictionary to hold the features
    # features = {
    #     'white_pawn': 0, 'black_pawn': 0, 'white_knight': 0, 'black_knight': 0,
    #     'white_bishop': 0, 'black_bishop': 0, 'white_rook': 0, 'black_rook': 0,
    #     'white_queen': 0, 'black_queen': 0, 'white_king': 0, 'black_king': 0
    # }

    # # mapping of pieces to features
    # piece_to_feature = {
    #     'P': 'white_pawn', 'p': 'black_pawn', 'N': 'white_knight', 'n': 'black_knight',
    #     'B': 'white_bishop', 'b': 'black_bishop', 'R': 'white_rook', 'r': 'black_rook',
    #     'Q': 'white_queen', 'q': 'black_queen', 'K': 'white_king', 'k': 'black_king'
    # }

    # count each piece
    # for char in piece_placement:
    #     if char in piece_to_feature:
    #         features[piece_to_feature[char]] += 1

    # at this point, actually we can also extract more useful information to be included
    # such as whose turn it is, castling rights and so on
    # but I choose to not use them to just make model simple
    # example could be features['turn'] = 1 if parts[1] == 'w' else 0 for determining the turn
    # in case we are all training from white side so never mind

    # return features

# def feature_hashing(df, column):
#     hasher = FeatureHasher(input_type='dict')
#     hashed_features = hasher.transform(df[column].apply(extract_features_from_fen))
#     return hashed_features

# def process_and_save_chunk(chunk, output_file):
#     hashed_features = feature_hashing(chunk, 'FEN')
#     hashed_features_df = pd.DataFrame(hashed_features.toarray())
#     chunk_processed = pd.concat([hashed_features_df, chunk], axis=1).drop('FEN', axis=1)

    # Save processed chunk
    # chunk_processed.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

# chunk_size = 1000
# input_file_path = 'Algorithm/evaluation_results_dimitar.csv'
# processed_file_path = 'Algorithm/processed_evaluation_results.csv'

# if not os.path.exists(processed_file_path):
#     print("Processed file not found. Starting feature extraction and processing...")
#     chunk_reader = pd.read_csv(input_file_path, chunksize=chunk_size)

#     for i, chunk in enumerate(chunk_reader):
#         print(f"Processing chunk {i + 1}...")
#         process_and_save_chunk(chunk, processed_file_path)
#         print(f"{(i + 1) * chunk_size} rows processed and saved.")

# else:
#     print("Loading processed data from the CSV file...")

# above is some try but did not make sense any more, it would be too time-comsuming




# reading and running the test model
# there is no tunning tecnique used so it is just a test
# to see how long it would take to simply run a regressor
print("Reading data...")
df = pd.read_csv('Algorithm/evaluation_results_dimitar.csv')

# drop uninterpretable FEN and unrelative Game Number
df = df.drop('FEN', axis=1)
df = df.drop('Game Number', axis=1)

# Convert 'Stockfish Evaluator (White)' column to numeric, including handling of + and - signs
df['Stockfish Evaluator (White)'] = pd.to_numeric(df['Stockfish Evaluator (White)'], errors='coerce')
print("Number of NaN values:", df['Stockfish Evaluator (White)'].isna().sum())
df = df.dropna(subset=['Stockfish Evaluator (White)'])

X = df.drop('Stockfish Evaluator (White)', axis=1)
Y = df['Stockfish Evaluator (White)']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print("Start training Random Forest Regressor...")
regressor = RandomForestRegressor(random_state=0)
regressor.fit(X_train, y_train)


# evaluate the model
y_pred = regressor.predict(X_test)

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: ", mae)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

# R^2 Score
r2 = r2_score(y_test, y_pred)
print("R^2 Score: ", r2)

# visulized result
# Actual vs Predicted values plot
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()


# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.axhline(y=0, color='k', linestyle='--')
plt.show()

#result of first run:
# Mean Absolute Error:  113.21851357112526
# Mean Squared Error:  42957.95720344254
# R^2 Score:  0.2916871991716854

# use Grid Search to tuning the hyperparameter
# which is the most regular way of tuning the model
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt']
# }

# grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=0), 
#                            param_grid=param_grid, 
#                            cv=3, 
#                            n_jobs=-1, 
#                            verbose=2)

# grid_search.fit(X_train, y_train)

# print("Best parameters:", grid_search.best_params_)

# best_regressor = grid_search.best_estimator_
# y_pred = best_regressor.predict(X_test)

# # Mean Absolute Error
# mae = mean_absolute_error(y_test, y_pred)
# print("Mean Absolute Error: ", mae)

# # Mean Squared Error
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error: ", mse)

# # R^2 Score
# r2 = r2_score(y_test, y_pred)
# print("R^2 Score: ", r2)

# # visulized result
# # Actual vs Predicted values plot
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs Predicted Values')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)  # Diagonal line
# plt.show()


# # Residual plot
# residuals = y_test - y_pred
# plt.scatter(y_pred, residuals)
# plt.xlabel('Predicted Values')
# plt.ylabel('Residuals')
# plt.title('Residuals vs Predicted Values')
# plt.axhline(y=0, color='k', linestyle='--')
# plt.show()

# Best parameters: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}
# Mean Absolute Error:  103.9378393255339
# Mean Squared Error:  34295.82021537535
# R^2 Score:  0.4018760608678794

# Save the best model from grid search
best_params = {
    'max_depth': 10, 
    'max_features': 'sqrt', 
    'min_samples_leaf': 4, 
    'min_samples_split': 2, 
    'n_estimators': 300
}


optimal_regressor = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    random_state=42
)

optimal_regressor.fit(X_train, y_train)
y_pred = optimal_regressor.predict(X_test)

model_filename = 'trained_random_forest_regressor.joblib'
joblib.dump(optimal_regressor, model_filename)
print(f"Model saved as {model_filename}")
