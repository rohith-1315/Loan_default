import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, 
    precision_score, confusion_matrix, 
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
data = pd.read_csv('Loan_default.csv')
data.fillna(data.median(numeric_only=True), inplace=True)
for column in data.select_dtypes(include='object').columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# Drop LoanID column
data = data.drop(columns=['LoanID'])

# Separate features and target
X = data.drop(columns=['Default'])
y = data['Default']

# Encode categorical variables
label_encoders = {}
for column in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Decision Tree model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print metrics
print(f"Model accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")

# Confusion Matrix Plot
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Prediction Function
def predict_loan_default():
    user_input = {}
    try:
        user_input['Age'] = int(input("Enter Age: "))
        if user_input['Age'] < 0:
            raise ValueError("Age must be non-negative!")

        user_input['Income'] = int(input("Enter Income: "))
        if user_input['Income'] < 0:
            raise ValueError("Income must be non-negative!")

        user_input['LoanAmount'] = int(input("Enter Loan Amount: "))
        if user_input['LoanAmount'] < 0:
            raise ValueError("Loan Amount must be non-negative!")

        user_input['CreditScore'] = int(input("Enter Credit Score (300-850): "))
        if not (300 <= user_input['CreditScore'] <= 850):
            raise ValueError("Credit Score must be between 300 and 850!")

        user_input['MonthsEmployed'] = int(input("Enter Months Employed: "))
        if user_input['MonthsEmployed'] < 0:
            raise ValueError("Months Employed must be non-negative!")

        user_input['NumCreditLines'] = int(input("Enter Number of Credit Lines: "))
        if user_input['NumCreditLines'] < 0:
            raise ValueError("Number of Credit Lines must be non-negative!")

        user_input['InterestRate'] = float(input("Enter Interest Rate: "))
        if user_input['InterestRate'] < 0:
            raise ValueError("Interest Rate must be non-negative!")

        user_input['LoanTerm'] = int(input("Enter Loan Term (in months): "))
        if user_input['LoanTerm'] <= 0:
            raise ValueError("Loan Term must be positive!")

        user_input['DTIRatio'] = float(input("Enter DTI Ratio: "))
        if user_input['DTIRatio'] < 0:
            raise ValueError("DTI Ratio must be non-negative!")

    except ValueError as e:
        print(f"Invalid input: {e}")
        return

    for column in label_encoders.keys():
        choices = list(label_encoders[column].classes_)
        print(f"Choose {column} from {choices}: ")
        user_choice = input()
        try:
            user_input[column] = label_encoders[column].transform([user_choice])[0]
        except ValueError:
            print(f"Invalid choice for {column}.")
            return

    user_df = pd.DataFrame([user_input])
    prediction = model.predict(user_df)
    if prediction[0] == 1:
        print("Prediction: The loan will likely default.")
    else:
        print("Prediction: The loan is unlikely to default.")

    try:
        choice = int(input("Do you want to enter another input? (1 for yes / 0 for no): "))
        if choice:
            predict_loan_default()
    except ValueError:
        print("Invalid choice.")

# Start the interactive prediction
predict_loan_default()
