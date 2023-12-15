import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

print("Reading data...")
df = pd.read_csv('Algorithm/evaluation_results_dimitar.csv')

# analyze the data to get bin group
df['Stockfish Evaluator (White)'] = pd.to_numeric(df['Stockfish Evaluator (White)'], errors='coerce')
df = df.dropna(subset=['Stockfish Evaluator (White)'])
numeric_description = df['Stockfish Evaluator (White)'].describe()
print(numeric_description)

# count    349832.000000
# mean         29.930933
# std         240.000263
# min       -8810.000000
# 25%         -23.000000
# 50%          29.000000
# 75%          84.000000
# max        8071.000000
# Name: Stockfish Evaluator (White), dtype: float64

bins = [-float('inf'), -1000, -100, -23, 84, 500, 1000, float('inf')]
labels = ['Extremely Negative', 'Highly Negative', 'Negative', 
          'Slightly Negative to Average', 'Positive', 
          'Highly Positive', 'Extremely Positive']

# Convert continuous values to categorical bins
df['Stockfish Evaluator (White) Binned'] = pd.cut(df['Stockfish Evaluator (White)'], bins=bins, labels=labels)
binned_distribution = df['Stockfish Evaluator (White) Binned'].value_counts()
print(binned_distribution)

# Slightly Negative to Average    175288
# Positive                         76288
# Highly Negative                  51115
# Negative                         36273
# Highly Positive                  10027
# Extremely Positive                 475
# Extremely Negative                 366
# Name: Stockfish Evaluator (White) Binned, dtype: int64


# training and running the model
print("Training Random Forest Classifier...")

# below is a regular model just for test, not tuning yet
X = df.drop(['Stockfish Evaluator (White)', 'Stockfish Evaluator (White) Binned', 'FEN'], axis=1)
Y = df['Stockfish Evaluator (White) Binned']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)


# evaluation of model
print("Evaluating the model...")
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=labels))
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Accuracy: 0.5926079437449083
# Classification Report:
#                               precision    recall  f1-score   support

#           Extremely Negative       0.56      0.18      0.28        82
#              Highly Negative       0.40      0.10      0.16       102
#                     Negative       0.56      0.58      0.57     10202
# Slightly Negative to Average       0.48      0.32      0.38      1982
#                     Positive       0.20      0.09      0.13      7243
#              Highly Positive       0.48      0.46      0.47     15143
#           Extremely Positive       0.68      0.77      0.72     35213

#                     accuracy                           0.59     69967
#                    macro avg       0.48      0.36      0.39     69967
#                 weighted avg       0.56      0.59      0.57     69967


# Use Grid Search tuning the model
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
print("Performing grid search...")
grid_search.fit(X_train, y_train)

print("Best parameters found by grid search:", grid_search.best_params_)

best_classifier = grid_search.best_estimator_
y_pred = best_classifier.predict(X_test)

# evaluate the best model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()