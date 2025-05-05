import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model #load model
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import seaborn as sns
import pickle #save encoder

# Read data to pandas dataframe
df = pd.read_csv('diabetes.csv')

# Count the number of null values in each column
print("Number of null values in each column:")
print(df.isnull().sum())

# Count the number of zero values in each column
print("Number of zero values in each column:")
print((df == 0).sum())

#Glucose, Insulin, skin thickenss, BMI and Blood Pressure datas can't be 0
# now replacing zero values with the mean of the column
df['BMI'] = df['BMI'].replace(0,df['BMI'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0,df['BloodPressure'].mean())
df['Glucose'] = df['Glucose'].replace(0,df['Glucose'].mean())
df['Insulin'] = df['Insulin'].replace(0,df['Insulin'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0,df['SkinThickness'].mean())

# Divide X and y
X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = df['Outcome']

# Split data into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# scaling the data
scaler_x= StandardScaler()
X_train=scaler_x.fit_transform(X_train)
X_test=scaler_x.transform(X_test)


######################
# ANN neural network #
######################
# Build and train ANN
model = Sequential()
model.add(Dense(8, input_dim=X.shape[1], kernel_initializer='normal', activation='relu')) # 8 size input layer
model.add(Dropout(0.1)) # To avoid overfitting
model.add(Dense(8, activation='relu')) # 8 size hidden layer
model.add(Dropout(0.1)) # To avoid overfitting
model.add(Dense(1, activation='sigmoid')) # 1 size output layer
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy','accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=8,  verbose=1, validation_data=(X_test,y_test))

# visualize training
print(history.history.keys())

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Predict with test data
y_pred = model.predict(X_test)
y_pred_class = y_pred > 0.5 # true / false

# Confusion Matrix and metrics
cm = confusion_matrix(y_test, y_pred_class)
acc = accuracy_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)

print(cm)
print (f'accuracy_score of ANN neural network: {acc}')
print (f'recall_score of ANN neural network: {recall}')
print (f'precision_score of ANN neural network: {precision}')

print (f'y_test: {y_test.value_counts()}')

sns.heatmap(cm, annot=True, fmt='g')
plt.show()

# Save model to disk
model.save('diabetes_model.h5')

# save scalers to disk
with open('diabetes-scaler_x.pickle', 'wb') as f:
    pickle.dump(scaler_x, f)

# load model
model = load_model('diabetes_model.h5')

# load scalers
with open('diabetes-scaler_x.pickle', 'rb') as f:
    scaler_x= pickle.load(f)

# predict with new data
Xnew = pd.read_csv('new_data_for_prediction.csv')
Xnew_org = Xnew
Xnew= scaler_x.transform(Xnew)
ynew= model.predict(Xnew)

# get scaled value back to unscaled
Xnew = scaler_x.inverse_transform(Xnew)

for i in range(len(ynew)):
    prediction = 1 if ynew[i][0] >= 0.5 else 0
    print(f'{Xnew_org.iloc[i]}\nPredicted Status (threshold 0.5): {prediction}\n')



#######################
# Logistic regression #
#######################
# Create model for logistic regression
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predicting outputs with X_test as inputs
y_pred = model.predict(X_test)

# Estimate the result by the confusion matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True)
plt.show()
print("confusion matrix of logistic regression: ", cm)

# Calculate accuracy score
acc = accuracy_score(y_test, y_pred)
print (f'accuracy score of logistic regression: {acc:.2f} ')

# Calculate precision score
precision = precision_score(y_test, y_pred)
print (f'precision score of logistic regression: {precision:.2f} ')

# Calculate recall score
recall = recall_score(y_test,y_pred)
print (f'recall score of logistic regression: {recall:.2f} ')

#new data
new_data = pd.read_csv('new_data_for_prediction.csv')

# predict with new data and create dataframe
new_y = pd.DataFrame(model.predict(new_data))

# apply species information based on the prediction
new_data['Outcome'] = new_y[0]
print(new_data)


