import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def _plot_line(x, y,ax):
    lr = LinearRegression()
    lr.fit(x.reshape(-1,1),y.reshape(-1,1))
    yhat = lr.predict(x.reshape(-1,1))
    sns.lineplot(x=x, y = yhat.ravel(), color = "red")

def plot_perfect_linear(minX=0, maxX=50, n=100, seed = 16, ax = None, fit_line = False):
    np.random.seed(seed)
    a = np.random.randint(1,10)
    b = np.random.randint(20,100)
    x = np.linspace(minX, maxX, n)
    y = a*x+b
    plot = sns.scatterplot(x=x,y=y, ax=ax)
    if fit_line:
        _plot_line(x,y,ax=ax)
    return plot




def plot_noisy_linear(minX=0, maxX=50, n=100, noise_strength = 1, seed = 16, ax = None, fit_line=False):
    np.random.seed(seed)
    a = np.random.randint(1,10)
    b = np.random.randint(20,100)
    x = np.linspace(minX, maxX, n)
    noise = np.random.normal(size = n) * (minX+maxX)/10 * noise_strength
    y = a*x+b+noise
    plot = sns.scatterplot(x=x,y=y, ax=ax)
    if fit_line:
        _plot_line(x,y,ax=ax)
    return plot