# 1. Libraries
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 2. Model
model = joblib.load(r'C:\Users\AI\PycharmProjects\Projects\Final_Project\our_model.pkl')

# 3. User Interface
st.title('Loan Approval Application 💰🏦')

age        = st.slider('Enter your age: ',                                 min_value=18, max_value=100, value=18, step=1)
emp_length = st.slider('Enter your employment length (years): ',           min_value=0,  max_value=50,  value=0,  step=1)
cred_hist_length = st.slider('Enter your credit history length (years): ', min_value=0,  max_value=30,  value=0,  step=1)

income              = st.number_input('Enter your income: ',     min_value=0,     value=0,     step=1_000)
loan_amount         = st.number_input('Enter the loan amount: ', min_value=1_000, value=1_000, step=1_000)
loan_int_rate       = st.selectbox('Enter your loan interest rate (%): ', [f'{i}%' for i in range(0, 41)])
loan_percent_income = round(loan_amount / income, 2) if income > 0 else 0

loan_intent            = st.radio('What is the purpose of this loan? ',    ('Personal',
                                                                            'Home Improvement',
                                                                            'Education',
                                                                            'Medical',
                                                                            'Venture',
                                                                            'Debt Consolidation'))
loan_grade             = st.radio('Select your loan grade: ',              ('A', 'B', 'C', 'D', 'E', 'F', 'G'))
home_ownership         = st.radio('Select your home ownership status: ',   ('Rent',
                                                                            'Mortgage',
                                                                            'Own',
                                                                            'Other'))
person_default_on_file = st.radio('Have you defaulted on a loan before? ', ('Yes', 'No'))

# 4. Preparing Our Data
train        = pd.read_csv(r'C:\Users\AI\PycharmProjects\Projects\Final_Project\train.csv')
train        = train.drop(columns=['id'])
invalid_rows = train[train['person_emp_length'] > train['person_age']]
train        = train.drop(invalid_rows.index)

X = train.drop('loan_status', axis=1)
y = train['loan_status']

X_mixed_encoded                              = X.copy()
X_mixed_encoded['cb_person_default_on_file'] = X_mixed_encoded['cb_person_default_on_file'].map({'N': 0, 'Y': 1})
X_mixed_encoded['loan_grade']                = X_mixed_encoded['loan_grade'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})

train_data_with_target                = X.copy()
train_data_with_target['loan_status'] = y

remaining_categorical_cols = X.select_dtypes(include='object').columns
remaining_categorical_cols = remaining_categorical_cols.difference(['cb_person_default_on_file', 'loan_grade'])

for col in remaining_categorical_cols:
    means                = train_data_with_target.groupby(col)['loan_status'].mean()
    X_mixed_encoded[col] = X_mixed_encoded[col].map(means)

numeric_columns = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length'
]

scaler                                    = StandardScaler()
X_mixed_encoded_Standard                  = X_mixed_encoded.copy()
X_mixed_encoded_Standard[numeric_columns] = scaler.fit_transform(X_mixed_encoded[numeric_columns])

loan_grade_encoded     = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}[loan_grade]
person_default_encoded = 1 if person_default_on_file == 'Yes' else 0

loan_intent_mapping = {
    'Personal'          : 'PERSONAL',
    'Home Improvement'  : 'HOMEIMPROVEMENT',
    'Education'         : 'EDUCATION',
    'Medical'           : 'MEDICAL',
    'Venture'           : 'VENTURE',
    'Debt Consolidation': 'DEBTCONSOLIDATION'
}

home_ownership_mapping = {
    'Rent'    : 'RENT',
    'Mortgage': 'MORTGAGE',
    'Own'     : 'OWN',
    'Other'   : 'OTHER'
}

loan_intent_mapped    = loan_intent_mapping[loan_intent]
home_ownership_mapped = home_ownership_mapping[home_ownership]

loan_intent_means    = train_data_with_target.groupby('loan_intent')['loan_status'].mean().to_dict()
home_ownership_means = train_data_with_target.groupby('person_home_ownership')['loan_status'].mean().to_dict()

loan_intent_encoded    = loan_intent_means[loan_intent_mapped]
home_ownership_encoded = home_ownership_means[home_ownership_mapped]

input_data = pd.DataFrame({
    'person_age'                : [age],
    'person_income'             : [income],
    'person_home_ownership'     : [home_ownership_encoded],
    'person_emp_length'         : [emp_length],
    'loan_intent'               : [loan_intent_encoded],
    'loan_grade'                : [loan_grade_encoded],
    'loan_amnt'                 : [loan_amount],
    'loan_int_rate'             : [float(loan_int_rate.strip('%'))],
    'loan_percent_income'       : [loan_percent_income],
    'cb_person_default_on_file' : [person_default_encoded],
    'cb_person_cred_hist_length': [cred_hist_length]
})

input_data_scaled                  = input_data.copy()
input_data_scaled[numeric_columns] = scaler.transform(input_data[numeric_columns])

# 5. Showing the result to the User
prediction_proba = model.predict_proba(input_data_scaled)[:, 1]

if prediction_proba < 0.5:
    st.success(f"Congratulations! Based on your input, there is a high chance (probability: {1 - prediction_proba[0]:.2f}) that your loan will be approved.")
    st.image(r"C:\Users\AI\PycharmProjects\Projects\Final_Project\happy.jpg", width=300)
else:
    st.error(f"Sorry, based on your input, there is a high chance (probability: {prediction_proba[0]:.2f}) that your loan will be rejected.")
    st.image(r"C:\Users\AI\PycharmProjects\Projects\Final_Project\sad.jpg", width=300)


# End of the Code...