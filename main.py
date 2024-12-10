import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_text, export_graphviz
import graphviz
import matplotlib.pyplot as plt
import io

# Load the dataset
data = pd.read_csv('bank.csv', delimiter=';')

# Encode categorical variables using one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Split the data into features (X) and target (y)
X = data_encoded.drop('y_yes', axis=1)  # Assuming 'y' is the target column
y = data_encoded['y_yes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model with pruning parameters
clf = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,  # Example: limit the depth of the tree
    min_samples_split=10,  # Example: minimum number of samples required to split an internal node
    min_samples_leaf=5,  # Example: minimum number of samples required to be at a leaf node
    max_leaf_nodes=20  # Example: maximum number of leaf nodes
)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Export the tree as text
tree_rules = export_text(clf, feature_names=list(X.columns))

# Visualize the decision tree
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=X.columns,  
                           class_names=['No', 'Yes'],  
                           filled=True, rounded=True,  
                           special_characters=True)  
graph = graphviz.Source(dot_data)
graph.render("tree")  # This will save the tree visualization as tree.pdf

# Capture the output of data.info() as a string
buffer = io.StringIO()
data.info(buf=buffer)
dataset_info = buffer.getvalue()

# Generate HTML report
html_template = """
<html>
<head><title>Decision Tree Classifier Report</title></head>
<body>
<h1>Decision Tree Classifier Report</h1>

<h2>Dataset Information</h2>
<pre>{dataset_info}</pre>

<h2>Model Evaluation</h2>
<p>Accuracy: {accuracy}</p>
<pre>{classification_report}</pre>

<h2>Decision Tree Rules</h2>
<pre>{tree_rules}</pre>

<h2>Decision Tree Visualization</h2>
<object data="tree.pdf" type="application/pdf" width="100%" height="600px">
    <embed src="tree.pdf" type="application/pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="tree.pdf">Download PDF</a>.</p>
    </embed>
</object>
</body>
</html>
"""

html_content = html_template.format(
    dataset_info=dataset_info,
    accuracy=accuracy,
    classification_report=pd.DataFrame(report).transpose().to_html(),
    tree_rules=tree_rules
)

# Save the HTML report
with open('decision_tree_report.html', 'w') as f:
    f.write(html_content)

print("Report saved as decision_tree_report.html")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import graphviz
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = '/content/bank-additional.csv'
data = pd.read_csv(file_path, sep=';')

# Data exploration (optional)
print("Dataset shape:", data.shape)
print("Columns:", data.columns)
print("Data types:\n", data.dtypes.value_counts())
print("\nInfo:")
data.info()
print("\nDuplicates:", data.duplicated().sum())
print("\nMissing values:\n", data.isna().sum())

# Separate features and target
X = data.drop(columns=['y'])
y = data['y']

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Handling missing values with SimpleImputer
numerical_imputer = SimpleImputer(strategy='median')
X[numerical_cols] = numerical_imputer.fit_transform(X[numerical_cols])

categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])

# Visualizations
# Histograms
X.hist(figsize=(12, 12), color='#cc5500')
plt.show()

# Count plots for categorical columns
for feature in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=feature, data=X, palette='Wistia')
    plt.title(f'Bar Plot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()

# Correlation heatmap
corr = X.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='Set3', linewidths=0.2)
plt.show()

# High correlation filter
high_corr_cols = ['emp.var.rate', 'euribor3m', 'nr.employed']
X = X.drop(columns=high_corr_cols)

# Update categorical and numerical columns after dropping high correlation columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Re-apply label encoding and imputation after dropping columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X[numerical_cols] = numerical_imputer.fit_transform(X[numerical_cols])
X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Visualize the decision tree
dot_data = export_graphviz(classifier, out_file=None,
                           feature_names=X.columns.tolist(),
                           class_names=list(map(str, classifier.classes_)),
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('decision_tree')  # This will save a PDF file of the decision tree

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation results
print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


# Function to predict whether a new customer will purchase a product or service
def predict_new_customer(new_customer_data):
    # Ensure the new data is in the same format as the training data
    for col in categorical_cols:
        if col in new_customer_data:
            new_customer_data[col] = label_encoders[col].transform([new_customer_data[col]])[0]

    # Create a DataFrame for the new customer data
    new_customer_df = pd.DataFrame([new_customer_data])

    # Handle missing values in the new customer data
    new_customer_df[numerical_cols] = numerical_imputer.transform(new_customer_df[numerical_cols])
    new_customer_df[categorical_cols] = categorical_imputer.transform(new_customer_df[categorical_cols])

    # Predict using the trained classifier
    prediction = classifier.predict_proba(new_customer_df)

    # Determine the prediction label
    if prediction[0][1] >= 0.5:  # Assuming 0.5 as the threshold for subscription probability
        return "Will subscribe to the product or service"
    else:
        return "Will not subscribe to the product or service"


# Example new customer data
new_customer_data = {
    'age': 35,
    'job': 'technician',
    'marital': 'single',
    'education': 'university.degree',
    'default': 'no',
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'month': 'aug',
    'day_of_week': 'thu',
    'duration': 210,
    'campaign': 1,
    'pdays': 999,
    'previous': 0,
    'poutcome': 'nonexistent',
    'cons.price.idx': 93.918,
    'cons.conf.idx': -42.7
}

# Remove high correlation columns from new customer data
for col in high_corr_cols:
    if col in new_customer_data:
        del new_customer_data[col]

# Predict whether the new customer will purchase a product or service
prediction = predict_new_customer(new_customer_data)
print("\nPrediction for new customer:", prediction)

# Display the decision tree
graph.view()
