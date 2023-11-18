import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('data/df_1_25.csv')

# drop month, day of month, day of week
data = data.drop(['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK'], axis=1)

# Split into training and test sets
y = data['DEP_DELAY']
X = data.drop(['DEP_DELAY'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# store the data
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)