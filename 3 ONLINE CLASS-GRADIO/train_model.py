
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

data = pd.read_csv("dataset.csv")

print("Dataset:")
print(data)

print("\nDataset shape:")
print(data.shape)


# --------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = data[
    [
        "StudyHours",
        "Attendance",
        "PreviousScore",
        "Assignments"
    ]
]

y = data["Marks"]


# --------------------------------------------------
# 3. SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. CREATE MODEL
# --------------------------------------------------

model = LinearRegression()


# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed.")


# --------------------------------------------------
# 6. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 7. EVALUATE MODEL
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print("\nModel Evaluation")
print("----------------")
print("MAE  :", mae)
print("RMSE :", rmse)
print("R2   :", r2)


# --------------------------------------------------
# 8. DISPLAY ACTUAL VS PREDICTED
# --------------------------------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(results)


# --------------------------------------------------
# 9. SAVE MODEL
# --------------------------------------------------

joblib.dump(model, "student_model.pkl")

print("\nModel saved as student_model.pkl")
