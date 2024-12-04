import sqlite3
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer, MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
import math

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    #connect to the data.db database
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    #joining the tables based on foreign keys
    query = """
    SELECT
    c.id AS client_id,
    c.age,
    c.job,
    c.marital,
    c.education,
    a.balance,
    a.in_default,
    a.housing,
    a.loan,
    ca.duration,
    ca.campaign,
    ca.pdays,
    ca.previous,
    o.poutcome,
    o.y
    FROM Clients c
    JOIN Accounts a ON c.id = a.client_id
    JOIN Campaigns ca ON a.id = ca.account_id
    JOIN Outcomes o ON ca.id = o.campaign_id;
    """

    # Execute the query and fetch the result into a DataFrame
    merged_df = pd.read_sql_query(query, conn)
    total_customers = len(merged_df)
    # Make basic filtering: 
    # 1. remove customers with negative balancce
    merged_df = merged_df[ merged_df["balance"] > 0 ]
    positive_balance_customers = len(merged_df)
    # 2 remove customers who are in default
    merged_df = merged_df[ merged_df["in_default"]  == "no" ]
    good_customers = len(merged_df)
    print("total", total_customers, 
          "positive", positive_balance_customers,
          "good customers", good_customers)

    train_and_test_df = merged_df[(merged_df.y == 'yes') | (merged_df.y =='no')]
    prediction_df = merged_df[merged_df.y=='unknown']
    prediction_df = prediction_df.drop("y", axis=1)

    return train_and_test_df, prediction_df


def total_return(num_reached, balances, alpha=0.1): 
    return -2 * num_reached + alpha * sum(balances)
# -2 * num_reached: This represents the cost of reaching out to num_reached individuals.
# calculates the total balance (or value) associated with the individuals reached.
# 
def avg_return(num_reached, balances, alpha=0.1):
    return total_return(num_reached, balances, alpha) / num_reached


def train_xgboost_model(xy_train):
    # Sample data creation (replace this with your actual DataFrame)
    np.random.seed(42)
    
    X, y = xy_train

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    # Preprocessing: Encode categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )

    # Create a pipeline that includes preprocessing and the classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])

    # Define hyperparameter grid for tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__subsample': [0.8, 1],
        'classifier__colsample_bytree': [0.8, 1],
    }

    # Set up GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2
    )

    # Fit the model using grid search
    grid_search.fit(X, y)

    # Output the best parameters
    print("Best parameters found: ", grid_search.best_params_)

    # Evaluate the best model on the validation set
    best_model = grid_search.best_estimator_
    return best_model


def evaluate_model(model, xy):
    X, y = xy

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    reached_customers = X.loc[y_pred == 1] 
    converted_customers = X.loc[(y_pred == 1) & (y == 1)]
    num_reached = len(reached_customers)
    num_converted = len(converted_customers)
    conversion_ratio = num_converted / num_reached
    expected_avg_return = avg_return(num_reached, reached_customers["balance"])
    actual_avg_return = avg_return(num_reached, converted_customers["balance"])
    return_ratio = actual_avg_return / expected_avg_return
    print(f'Accuracy: {accuracy:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f"Number of reached customers", num_reached)
    print(f"Number of converted customers", num_converted)
    print(f"Expected avg return", expected_avg_return)
    print(f"Actual avg return", actual_avg_return)
    print("Conversion ratio", conversion_ratio)
    print("Avg return ratio", return_ratio)
    return return_ratio


def split_dataset(df):
    # Separate features and target variable
    X = df[df.columns.drop(["client_id", "y"])]
    y = df["y"] == "yes"
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, y_train), (X_val, y_val)


def evaluate_predictions(X_future, y_future, validation_return_ratio):
    future_reached_customers = X_future.loc[y_future == 1]
    future_num_reached = len(future_reached_customers)
    future_expected_avg_return = avg_return(future_num_reached, future_reached_customers["balance"])
    future_expected_total_return = future_expected_avg_return * future_num_reached
    print(f"Number of reached customers", future_num_reached)
    print(f"Expected avg return", future_expected_avg_return * validation_return_ratio)
    print(f"Expected total return", future_expected_total_return * validation_return_ratio)



if __name__ == "__main__":
    print("Loading data")
    train_and_test_df, prediction_df = load_data()
    xy_train, xy_test = split_dataset(train_and_test_df)

    print("Training model")
    best_model = train_xgboost_model(xy_train)

    print("")
    print("Evaluation on train set")
    evaluate_model(best_model, xy_train)

    print("")
    print("Evaluation on test set")
    validation_return_ratio = evaluate_model(best_model, xy_test)

    print("")
    print("Evaluation on future prediction set")
    X_future = prediction_df.drop("client_id", axis=1)
    y_future = best_model.predict(X_future)
    evaluate_predictions(X_future, y_future, validation_return_ratio)
