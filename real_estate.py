import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

real_state_data = pd.read_csv('Real_Estate.csv')

from ydata_profiling import ProfileReport
prof = ProfileReport(real_state_data)
prof.to_file(output_file = 'Real Estate Price Prediction.html')

# Droping unnecessary columns
real_state_data.drop(columns = ['Transaction date','House age'], inplace =True)
real_state_data.head(2)

# Step 1 -> train/test/split
X_train,X_test,y_train,y_test = train_test_split(real_state_data.drop(columns=['House price of unit area']),
                                                 real_state_data['House price of unit area'],
                                                 test_size=0.2,
                                                random_state=42)


# Model initialization
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)


# Making predictions using the linear regression model
y_pred_lr = model.predict(X_test)

# Visualization: Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted House Prices')
plt.show()
