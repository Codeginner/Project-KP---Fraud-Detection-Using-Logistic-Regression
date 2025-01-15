import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Disable the warning for deprecated use of pyplot global object
#st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset
df = pd.read_csv("C:\python\projek kp\creditcard.csv")

# Preprocess the dataset
rbs = RobustScaler()
df_small = df[['Time', 'Amount']]
df_small = pd.DataFrame(rbs.fit_transform(df_small))
df_small.columns = ['scaled_time', 'scaled_amount']
df = pd.concat([df, df_small], axis=1)
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Create Streamlit app
st.title("Credit Card Fraud Detection")

# Display the dataset summary
st.subheader("Dataset Summary")
st.dataframe(df.describe())

# Display jointplot of 'scaled_amount' vs. 'Class'
st.subheader("Jointplot")
jointplot = sns.jointplot(x='scaled_amount', y='Class', data=df)
st.pyplot(jointplot.fig)

# Display countplot of original dataset 'Class'
st.subheader("Original Class Distribution")
class_order = [0, 1]
fig, ax = plt.subplots()
sns.countplot(x='Class', data=df, order=class_order, ax=ax)
st.pyplot(fig)

# Create a balanced dataset with 50/50 Class distribution
non_fraud = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]
non_fraud = non_fraud.sample(frac=1)
non_fraud = non_fraud[:492]
new_df = pd.concat([non_fraud, fraud])
new_df = new_df.sample(frac=1)

# create sns countplot for balanced Class distribution
st.subheader("Balanced Class Distribution")
new_class_order = [0, 1]
sns.countplot(x='Class', data=new_df, order=new_class_order)
st.pyplot()

# Prepare data for training
X_train = new_df.drop('Class', axis=1)
y_train = new_df['Class']

# Inisialisasi RobustScaler di luar fungsi
#rbs = RobustScaler()
#rbs.fit(X_train)

# Save and load the trained model to a file
model_filename = "C:\python\projek kp\credit_card_fraud_model.pkl"

model = LogisticRegression()
model.fit(X_train, y_train)
joblib.dump(model, "credit_card_fraud_model.pkl")
model = joblib.load("credit_card_fraud_model.pkl")

st.subheader("Balanced Classification Report")
# Perform prediction on the test set
pred_test = model.predict(X_train)
balanced_report = classification_report(y_train, pred_test, output_dict=True)
st.text(classification_report(y_train, pred_test))

st.subheader("Confusion Matrix")
st.text(confusion_matrix(y_train, pred_test))

st.subheader("Fraud Transactions")
recall_fraud = balanced_report.get('1', {}).get('recall')
if recall_fraud is not None:
    st.text(f"Percentage: {recall_fraud * 100:.2f}%")
else:
    st.text("Percentage for Fraud Transactions is not available.")

st.subheader("Non-Fraud Transactions")
recall_non_fraud = balanced_report.get('0', {}).get('recall')
if recall_non_fraud is not None:
    st.text(f"Percentage: {recall_non_fraud * 100:.2f}%")
else:
    st.text("Percentage for Non-Fraud Transactions is not available.")

st.subheader("Accuracy")
accuracy = round(accuracy_score(y_train, pred_test) * 100, 2)
st.text(f"Accuracy is --> {accuracy}%")

# Revisi kode untuk mengecek transaksi fraud dengan threshold tertentu
threshold = 0.5  # Ubah threshold sesuai kebutuhan (range 0 hingga 1)
fraud_prob_threshold = 0.9  # Ubah threshold probabilitas fraud sesuai kebutuhan (range 0 hingga 1)

# Add a feature to input credit card transaction for prediction
st.title("Credit Card Fraud Detection Model")
input_df = st.text_input('Check Credit Card Transaction')
input_df_lst = input_df.split(',')

if st.button("Check Transaction"):
    if len(input_df_lst) != len(X_train.columns):
        st.write("Invalid number of features. Please enter {} features.".format(len(X_train.columns)))
    else:
        # Preprocess the input data
        features = np.array(input_df_lst, dtype=np.float64)
        #features = rbs.transform(features.reshape(1, -1))

        # Perform prediction on the entered credit card transaction
        prediction = model.predict(features.reshape(1, -1))
        probabilities = model.predict_proba(features.reshape(1, -1))
        fraud_prob = probabilities[0][1]  # Probabilitas fraud (class 1)
        #prediction = model.predict(features)
        #probabilities = model.predict_proba(features)

        if prediction == 0:
            st.write("The transaction is not suspicious.")
        else:
            st.write("The transaction is suspicious.")

        # Display the probabilities for each class
        st.write("Probability of being Non-Fraud : {:.2f}%".format(probabilities[0][0] * 100))
        st.write("Probability of being Fraud : {:.2f}%".format(probabilities[0][1] * 100))

        # Calculate accuracy on training data
        accuracy = accuracy_score(y_train, pred_test)
        st.write("Accuracy: {:.2f}%".format(accuracy * 100))

        # Check if the prediction is above the fraud probability threshold
        if fraud_prob > fraud_prob_threshold:
            st.write("The transaction is highly likely to be fraud.")
        else:
            st.write("The transaction is not highly likely to be fraud.")

        # Plot Logistic Regression curve
        x_vals = np.linspace(-10, 10, 1000)
        y_vals = 1 / (1 + np.exp(-x_vals))
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(x_vals, y_vals)
        plt.xlabel("z (Logit)")
        plt.ylabel("Probability")
        plt.title("Logistic Regression Curve")
        plt.axvline(x=np.dot(features, model.coef_[0]) + model.intercept_, color='r', linestyle='--', label='Logit(z)')
        plt.axhline(y=probabilities[0][1], color='g', linestyle='--', label='Probability of being Fraud')
        plt.axhline(y=probabilities[0][0], color='b', linestyle='--', label='Probability of being Non-Fraud')
        plt.legend()
        st.pyplot(fig)
