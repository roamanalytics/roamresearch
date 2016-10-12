"""
Companion code for the post 'Getting into the middle of regression'.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

__author__ = 'Nick Dingwall and Adam Foster'

matplotlib.style.use('../../src/roam.mplstyle')

POINT_COLOR = '#0499CC'
LINE_COLORS = ['#FDBA58','#F44E42']

def plot_scatter_and_line(ax, x, y, x_line, y_line, x_end, y_end, title,
                          color=0):
    # Set up the axes and plot them
    ax.set_xlim((-x_end, x_end))
    ax.set_ylim((-y_end, y_end))
    ax.plot((-x_end, x_end), (0, 0), 'k-')
    ax.plot((0, 0), (-y_end, y_end), 'k-')

    #Scatter
    ax.scatter(x, y, marker='x', color=POINT_COLOR)
    
    #ax.grid(True, which='minor', linestyle='--')
    #ax.minorticks_on()
    
    # Plot a red line
    ax.plot(x_line, y_line, color=LINE_COLORS[color], linewidth=2)
    ax.set_title(title)

def scatter_plus_regression(ax, x, y, x_end, y_end, model, title, color=0):
    
    # Build the model
    model.fit(x.reshape(-1, 1), y)
    
    # Getting the predictions
    x_line = np.array([-x_end, x_end]).T.reshape(-1, 1)
    y_line = model.predict(x_line)
    #y_end = max(y_end, np.ceil(np.max(np.abs(y_line))))
    
    plot_scatter_and_line(ax, x, y, x_line, y_line, x_end, y_end, title,
                          color=color)

def compare_normalizations(x, z, y, title1, title2):
    x_end = np.ceil(max([np.max(np.abs(i)) for i in [x, z]]))
    y_end = np.ceil(np.max(np.abs(y)))
    lr = LinearRegression()
    fig, ax = plt.subplots(2,1, figsize=(10,8))
    scatter_plus_regression(ax[0], x, y, x_end, y_end, lr, title1)
    scatter_plus_regression(ax[1], z, y, x_end, y_end, lr, title2, color=1)
    plt.show()

def transformed_scatter_plus_regression(ax, x, y, x_end, y_end, transform,
                                        model, title, color=0):
    # transformed data
    x_trans = transform(x)
    x_trans_end = transform(x_end)
    
    # train a model with x_trans
    model.fit(x_trans.reshape(-1, 1), y)
    
    # predict using transformed model    
    x_to_predict = np.array([-x_trans_end, x_trans_end]).T.reshape(-1, 1)
    x_to_plot = np.array([-x_end, x_end]).T.reshape(-1, 1)
    y_line = model.predict(x_to_predict)
    
    plot_scatter_and_line(ax, x, y, x_to_plot, y_line, x_end, y_end, title,
                          color=color)
    
def compare_transformed_normalizations(x, y, transform, model, title1, title2):
    x_end = np.ceil(max([np.max(np.abs(i)) for i in x]))
    y_end = np.ceil(np.max(np.abs(y)))

    fig, ax = plt.subplots(2,1, figsize=(10,8))
    transformed_scatter_plus_regression(
        ax[0], x, y, x_end, y_end, (lambda x : x), model, title1)
    transformed_scatter_plus_regression(
        ax[1], x, y, x_end, y_end, transform, model, title2, color=1)
    plt.show()
