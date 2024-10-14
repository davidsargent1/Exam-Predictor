import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay


df = pd.read_csv('Student Success.csv') # Converts csv to pandas dataframe
encoder = LabelEncoder() # Used to convert categorical data into numerical
passing_score = int(input('Choose a passing score:\n'))


# Replaces missing data with the mode of each column
def replace_null_vals(series_name):
    mode = df[series_name].mode()[0]
    df[series_name].fillna(mode, inplace=True)

# Converts categorical data into numerical data with Label Encoding
def encode_vals(series, series_name):
    if series.dtype != 'int64': 
        df[series_name] = encoder.fit_transform(df[series_name])


# Cleans up data of every column in the dataframe
for series_name, series in df.items(): 
    replace_null_vals(series_name)
    encode_vals(series, series_name)

# IMPORTANT: Converts scores above or equal to 60 to 1, and other scores to 0
df['Exam_Score'] = df['Exam_Score'].apply(lambda score: 1 if score >= passing_score else 0)
   
df.info() # Displays dataframe information in terminal

# Split into training and test models
x = df.drop('Exam_Score', axis=1) # x-axis is made up of all columns but 'Exam_Score'
y = df['Exam_Score'] # y-axis is Exam Score (dependent variable)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

model = LogisticRegression() # Sets up model
model.fit(x_train, y_train) # Trains model

# Makes predictions
y_pred = model.predict(x_test)

# Evaluates the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classif_report = classification_report(y_test, y_pred)

# Makes new dataframe to sort values by their effect on Exam_Score
feature_importance = pd.DataFrame({
    'Feature': x_train.columns,
    'Coefficient': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

# Displays the classification report, accuracy, top predictors
print("\nClassification Report:")
print(classif_report)
print(f"Accuracy: {(accuracy * 100):.2f}%")
print('----------------------------------------------')
print('\nTOP PREDICTORS:')
print('\n', feature_importance.head(19))  # Adjust the number to see more top predictors

# Graph Logistic Regression Results
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()