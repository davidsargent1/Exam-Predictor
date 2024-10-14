import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv('Student Success.csv') # Converts csv to pandas dataframe
encoder = LabelEncoder() # Used to convert categorical data into numerical

# Replaces missing data with the mode of each column
def replace_null_vals(series_name):
    mode = df[series_name].mode()[0]
    df[series_name].fillna(mode, inplace=True)

# Converts categorical data into numerical data with Label Encoding
def encode_vals(series, series_name):
    if series.dtype != 'int64': 
        df[series_name] = encoder.fit_transform(df[series_name])

def get_user_input(): # Gets a list of plots the user wants to see.
    plots = []
    user_input = ''

    while user_input != 'DONE': # Will stop asking for user input once DONE is entered.
        if user_input == 'LinReg': # Entering LinReg automatically generate Linear Regression plot.
            return plots
        user_input = (input('''
            \nWhat column would you like to compare to Exam_Score? Remember to use proper capitalization.
Enter LinReg to see the regression model. Enter DONE to stop entering columns.\n'''))
        plots.append(user_input)

    plots.pop() # Removes 'DONE' from plots list so that there is no error
    return plots

# Accepts list of columns to compare to Exam_Score
def create_plots(plot_types):
    try:
        if plot_types.count('LinReg') > 0: # Prioritizes Linear Regression, so if the user wants it, the plot will always be this.
            # Graph Linear Regression Results
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_test, y=y_pred)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45-degree reference line
            plt.xlabel('Actual Exam Scores')
            plt.ylabel('Predicted Exam Scores')
            plt.title('Predicted vs. Actual Exam Scores')
            plt.show()

            for column in plot_types:    
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=df[column], y=df['Exam_Score'])
                plt.xlabel(column)
                plt.ylabel('Actual Exam Scores')
                plt.title(f'Actual Exam Scores vs. {column}')
                plt.show()

    except: # Executes in the event of a user entering unknown values 
        print('''\n\nWe were not able to recognize your selection. Maybe you misspelled a word?
Example: Enter \'Distance_From_Home\' to compare Distance From Home to Exam Score!''')

        comparison_plots = get_user_input()
        create_plots(comparison_plots)


# Cleans up data of every column in the dataframe
for series_name, series in df.items(): 
    replace_null_vals(series_name)
    encode_vals(series, series_name)
   
df.info() # Displays dataframe information in terminal

x = df.drop('Exam_Score', axis=1) # x-axis is made up of all columns but 'Exam_Score'
y = df['Exam_Score'] # y-axis is Exam Score (dependent variable)

# Split into training and test models
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression() # Sets up model
model.fit(x_train, y_train) # Trains model

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Output results
print(f"\nR-Squared Value: {r2}")
print(f"Mean Absolute Error: {mae}")

# Graph as many plots as wanted 
comparison_plots = get_user_input() # List of columns to compare to Exam_Score
create_plots(comparison_plots)