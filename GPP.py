import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.svm import SVR


def split_data(file): 
    '''we change date into a date right format, the prices as float, replacing commas with dots,
    and adding a index'''

    df = pd.read_csv(file, parse_dates=["Date"], dayfirst=True) 
    df["Close"] = (df["Close"].astype(str).str.replace(",", ".", regex=False).astype(float))
    df = df.sort_values("Date").reset_index(drop=True)
    df["DateIndex"] = df.index

    return df

def train_models(df):
    X = df["DateIndex"].values.reshape(-1, 1)
    y = df["Close"].values

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_lin.fit(X, y)
    svr_poly.fit(X, y)
    svr_rbf.fit(X, y)

    return svr_rbf, svr_lin, svr_poly

def plot_models(df, models):
    svr_rbf, svr_lin, svr_poly = models

    X = df["DateIndex"].values.reshape(-1, 1)
    y = df["Close"].values

    plt.scatter(X, y, label="Data")
    plt.plot(X, svr_rbf.predict(X), label="RBF model")
    plt.plot(X, svr_lin.predict(X), label="Linear model")
    plt.plot(X, svr_poly.predict(X), label="Polynomial model")

    plt.xlabel("Time Index")
    plt.ylabel("Price")
    plt.title("Support Vector Regression")
    plt.legend()
    plt.show()


def predict_price(models, x_index):
    svr_rbf, svr_lin, svr_poly = models

    X = np.array(x_index).reshape(-1,1)

    return {
        "rbf": svr_rbf.predict(X)[0],
        "linear": svr_lin.predict(X)[0],
        "poly": svr_poly.predict(X)[0]
    }

'''And the magic goes here, we call the functions'''
df = split_data('dataset.csv') # you should jus put your dataset here, but consider it goes on the standard google finance format
models = train_models(df)
plot_models(df, models)

'''Showing predictions'''

last_index = df["DateIndex"].iloc[-1]
print("Predicción último día real:", predict_price(models, last_index))

future_indices = list(range(last_index + 1, last_index + 6))
future_indices = np.array(future_indices).reshape(-1, 1)
print("Predicciones próximos 5 días:", predict_price(models, future_indices))

'''Graphic of predictions vs real'''

svr_rbf, svr_lin, svr_poly = models
X = df["DateIndex"].values.reshape(-1,1)
y = df["Close"].values

plt.scatter(X, y, label="Datos reales", color='black')
plt.plot(X, svr_rbf.predict(X), label='RBF model', color='red')
plt.plot(X, svr_lin.predict(X), label='Linear model', color='green')
plt.plot(X, svr_poly.predict(X), label='Polynomial model', color='blue')

plt.plot(future_indices, svr_rbf.predict(future_indices), 'r--', label='RBF future')
plt.plot(future_indices, svr_lin.predict(future_indices), 'g--', label='Linear future')
plt.plot(future_indices, svr_poly.predict(future_indices), 'b--', label='Poly future')

plt.xlabel("Time Index")
plt.ylabel("Price")
plt.title("SVR Prediction")
plt.legend()
plt.show()