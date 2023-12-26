import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from fpdf import FPDF  # You may need to install this library using: pip install fpdf

def load_data(file_path):
    raw_data = pd.read_csv(file_path)
    return raw_data

def clean_and_engineer_data(raw_data):
    clean_data = raw_data.copy()
    
    # Format column names to be lower-case
    new_col_names = [col.lower() for col in clean_data.columns]
    clean_data.columns = new_col_names

    # Fill null values - 'item_weight' using mean
    clean_data['item_weight'].fillna(clean_data['item_weight'].mean(), inplace=True)

    # Fill null values - 'outlet_size' using mode per outlet_type
    outlet_size_mode_pt = clean_data.pivot_table(values='outlet_size', columns='outlet_type', aggfunc=lambda x: x.mode())
    missing_values = clean_data['outlet_size'].isnull()
    clean_data.loc[missing_values, 'outlet_size'] = clean_data.loc[missing_values, 'outlet_type'].apply(
        lambda x: outlet_size_mode_pt[x].outlet_size)

    # Replace 0s with the mean - 'item_visibility'
    clean_data.loc[:,'item_visibility'].replace(to_replace=0,
                                                value=clean_data['item_visibility'].mean(),
                                                inplace=True)

    # Replace repetitive values - 'item_fat_content'
    clean_data['item_fat_content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}, inplace=True)

    # Create a new 'item_category' feature
    clean_data['item_category'] = clean_data['item_identifier'].apply(lambda x: x[:2])
    clean_data['item_category'] = clean_data['item_category'].replace({'FD': 'Food', 'DR': 'Drink', 'NC': 'Non-Consumable'})

    # Update 'item_fat_content' for Non-Consumables
    clean_data.loc[clean_data['item_category'] == 'Non-Consumable', 'item_fat_content'] = 'No Edible'

    # Create a new 'outlet_years' feature
    clean_data['outlet_years'] = 2013 - clean_data['outlet_establishment_year']

    return clean_data

def visualize_data(clean_data):
    # Plot categorical features
    plt.figure(figsize=(5,5))
    sns.countplot(x='item_fat_content', data=clean_data)
    plt.show()

    
    # plot outlet_size
    plt.figure(figsize=(5,5))
    sns.countplot(x='outlet_size', data=clean_data)
    plt.show()


    # plot outlet_location_type
    plt.figure(figsize=(5,5))
    sns.countplot(x='outlet_location_type', data=clean_data)
    plt.show()


    # plot item_category
    plt.figure(figsize=(5,5))
    sns.countplot(x='item_category', data=clean_data)
    plt.show()
    
    # outlet_establishment_year column count
    plt.figure(figsize=(6,6))
    sns.countplot(x='outlet_establishment_year', data=clean_data)
    plt.show()

    # outlet_years column count
    plt.figure(figsize=(6,6))
    sns.countplot(x='outlet_years', data=clean_data)
    plt.show()
    
    # Data distribution of numerical values
    clean_data.hist(figsize=(12,8))
    plt.show()

def preprocess_data(clean_data):
    # Apply label encoding to some features
    encoder = LabelEncoder()
    cols_to_encode = ['item_identifier', 'item_type', 'outlet_identifier']

    for col in cols_to_encode:
        clean_data[col] = encoder.fit_transform(clean_data[col])

    # Apply one-hot encoding to some features
    clean_data = pd.get_dummies(clean_data, columns=['item_fat_content', 'outlet_size', 'outlet_location_type',
                                                     'outlet_type', 'item_category'])

    return clean_data

def train_models_and_evaluate(X, y):
    # df to store model error and scores
    model_scores = pd.DataFrame(columns=['model', 'rmse', 'r2_score'])

    def train_and_evaluate_model(model_name, model, X, y):
        # split the data
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=42)

        # create a training pipeline
        pipeline = make_pipeline(StandardScaler(), model)

        # apply scaling on training data and train the model
        pipeline.fit(X_train, y_train)

        # predict the validation set
        y_hat = pipeline.predict(X_validate)

        # evaluate the model
        rmse = np.sqrt(mean_squared_error(y_validate, y_hat))
        model_score = r2_score(y_validate, y_hat)

        # adding error and score, to the scores dataframe
        model_scores.loc[len(model_scores)] = [model_name, rmse, model_score]

        print('----------------------------------')
        print(model_name, ' Report:')
        print('----------------------------------')
        print('RMSE: ', rmse)
        print('R2 Score: ', model_score)

    # Linear Regression
    linear_regression_model = LinearRegression()
    train_and_evaluate_model('Linear Regression', linear_regression_model, X, y)

    # Ridge Regularization
    ridge_model = Ridge()
    train_and_evaluate_model('Ridge', ridge_model, X, y)

    # Lasso Regularization
    lasso_model = Lasso()
    train_and_evaluate_model('Lasso', lasso_model, X, y)

    # SVM
    svr_model = SVR()
    train_and_evaluate_model('SVM', svr_model, X, y)

    # Decision Tree
    dtr_model = DecisionTreeRegressor()
    train_and_evaluate_model('Decision Tree', dtr_model, X, y)

    # Random Forest
    rfr_model = RandomForestRegressor()
    train_and_evaluate_model('Random Forest', rfr_model, X, y)

    # XGBoost
    xgbr_model = XGBRegressor()
    train_and_evaluate_model('XGBoost', xgbr_model, X, y)

    return model_scores

def train_final_model(X, y):
    # Train a Linear Regression model with all the data
    model_pipeline = make_pipeline(StandardScaler(), LinearRegression())
    model_pipeline.fit(X, y)
    return model_pipeline

def prepare_test_data(model, test_data):
    # IMPORTANT: The feature labels that were added during the training process must be added
    # adding missing features
    test_data['item_fat_content_No Edible'] = 0
    test_data['item_fat_content_Regular'] = 0
    test_data['outlet_size_High'] = 0
    test_data['item_category_Drink'] = 0
    test_data['item_category_Non-Consumable'] = 0

    # drop unnecessary features
    test_data = test_data.drop(columns=['outlet_establishment_year'])

    # IMPORTANT: The features must match the same order as the training data
    # re-order columns
    test_data = test_data[['item_identifier',
                           'item_weight',
                           'item_visibility',
                           'item_type',
                           'item_mrp',
                           'outlet_identifier',
                           'outlet_years',
                           'item_fat_content_Low Fat',
                           'item_fat_content_No Edible',
                           'item_fat_content_Regular',
                           'outlet_size_High',
                           'outlet_size_Medium',
                           'outlet_size_Small',
                           'outlet_location_type_Tier 1',
                           'outlet_location_type_Tier 2',
                           'outlet_location_type_Tier 3',
                           'outlet_type_Grocery Store',
                           'outlet_type_Supermarket Type1',
                           'outlet_type_Supermarket Type2',
                           'outlet_type_Supermarket Type3',
                           'item_category_Drink',
                           'item_category_Food',
                           'item_category_Non-Consumable']]

    # Predict the testing data
    y_hat = model.predict(test_data)
    
    # Create a new DataFrame with the results
    results = test_data[['item_identifier', 'outlet_identifier']]
    results['prediction'] = y_hat

    return results

def generate_pdf_report(model_scores, results):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Model Evaluation Report', 0, 1, 'C')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(4)

        def chapter_body(self, body):
            self.set_font('Arial', '', 12)
            self.multi_cell(0, 10, body)
            self.ln()

    pdf = PDF()

    pdf.add_page()

    # Model Evaluation Section
    pdf.chapter_title('Model Evaluation')
    for index, row in model_scores.iterrows():
        pdf.chapter_body(f"Model: {row['model']}\nRMSE: {row['rmse']}\nR2 Score: {row['r2_score']}\n")

    # Test Results Section
    pdf.chapter_title('Test Results')
    pdf.chapter_body(results.to_string())

    pdf.output("model_evaluation_report.pdf")


def main(test_file_path,train_file_path):
    raw_train_data = load_data(train_file_path)
    clean_train_data = clean_and_engineer_data(raw_train_data)
    visualize_data(clean_train_data)
    preprocessed_train_data = preprocess_data(clean_train_data)
    model_scores = train_models_and_evaluate(preprocessed_train_data.drop(columns=['outlet_establishment_year', 'item_outlet_sales']), preprocessed_train_data['item_outlet_sales'])
    final_model = train_final_model(preprocessed_train_data.drop(columns=['outlet_establishment_year', 'item_outlet_sales']), preprocessed_train_data['item_outlet_sales'])

    raw_test_data = load_data(test_file_path)
    clean_test_data = clean_and_engineer_data(raw_test_data)
    preprocessed_test_data = preprocess_data(clean_test_data)
    test_results = prepare_test_data(final_model, preprocessed_test_data)

    generate_pdf_report(model_scores, test_results)


# Example usage:
train_file_path = 'train.csv'
test_file_path = 'test.csv'
main(test_file_path,train_file_path)