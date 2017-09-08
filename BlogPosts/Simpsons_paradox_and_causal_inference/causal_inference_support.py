import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from IPython.display import display, SVG, Image
plt.style.use('../../src/roam.mplstyle')
np.random.seed(1)

# Plotters
def side_by_side(ax, v1, v2, title, xlab, ylab, col1='blue', col2='green'):
    x1 = [1]*len(v1)
    x2 = [2]*len(v2)
    ax.plot(x1, v1, 'x', color=col1)
    ax.plot(x2, v2, 'x', color=col2)
    ax.set_title(title, fontsize=18)
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

def coloured_by_group(ax, v1, v2, c1, c2, title):
    x1 = [1]*len(v1)
    x2 = [2]*len(v2)
    ax.scatter(x1, v1, 50, marker='x', color=c1)
    ax.scatter(x2, v2, 50, marker='x', color=c2)
    ax.set_title(title, fontsize=18)
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])

def two_boxplot(ax, d1, d2, title, xlab='Treatment', ylab='Response'):
    ax.boxplot([d1, d2])
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

def data_generator1():
    while True:
        group = np.random.binomial(1, 0.5)
        treatment = np.random.binomial(1, 1/8 if group else 7/8)
        response = np.random.normal( 2 + 2*treatment + 4*group, 2)
        group =  "Urban" if group else "Rural"
        treatment = 2 if treatment else 1
        yield dict(location=group, treatment=treatment, response=response)

def data_generator2():
    while True:
        treatment = np.random.binomial(1, 0.5)
        response = np.random.normal( 5 + 0.5*treatment, 1)
        if treatment == 0:
            lifestyle = response > 6
        else:
            lifestyle = response > 4.5
        treatment = 2 if treatment else 1
        yield dict(treatment=treatment, response=response, scuba_diving=lifestyle)

def data_generator3():
    while True:
        doctor = np.random.binomial(1, 0.5)
        patient = np.random.binomial(1, 0.5)
        dostoyevsky = doctor and patient
        treatment = np.random.binomial(1, 0.8 if doctor else 0.2)
        response = np.random.normal( 5 + 0.5*treatment - patient, 1)

        # For a nicer output
        treatment = 2 if treatment else 1
        yield dict(doctor=doctor, patient=patient, dostoyevsky=dostoyevsky,
                   treatment=treatment, response=response)

def get_dataframe(generator, n):
    data= [next(generator) for x in range(n)]
    return pd.DataFrame(data)


def run_ttest(df, variable, level1, level2, response):
    v1 = df[df[variable] == level1][response]
    v2 = df[df[variable] == level2][response]
    test=stats.ttest_ind(v1,v2)
    print("Performing t-test assuming equal variance: p-value is {:.2}".format(
        test.pvalue))


def print_table(table):
    df = pd.DataFrame.from_items(table)
    print(df.to_string(index=False))

def match(group_00, group_01, group_10, group_11):
    a1, a2, b1, b2 = list(group_00), list(group_01), list(group_10), list(group_11)
    size_a= min(len(a1), len(a2))
    a1, a2 = a1[:size_a], a2[:size_a]
    size_b = min(len(b1), len(b2))
    b1, b2 = b1[:size_b], b2[:size_b]
    return a1+b1, a2+b2

def plot_match(ax, group_00, group_01, group_10, group_11, title_l, title_r, xlab, ylab, col):
    a1, a2, b1, b2 = list(group_00), list(group_01), list(group_10), list(group_11)
    size_a= min(len(a1), len(a2))
    size_b = min(len(b1), len(b2))
    side_by_side(ax[0], a1[:size_a], a2[:size_b], title_l, xlab, ylab)
    side_by_side(ax[1], b1[:size_b], b2[:size_b], title_r, xlab, ylab)
    ua1 = a1[size_a:]
    ua2 = a2[size_a:]
    ca1 = [col for _ in ua1]
    ca2 = [col for _ in ua2]
    coloured_by_group(ax[0], ua1, ua2, ca1, ca2, title_l)
    ub1 = b1[size_b:]
    ub2 = b2[size_b:]
    cb1 = [col for _ in ub1]
    cb2 = [col for _ in ub2]
    coloured_by_group(ax[1], ub1, ub2, cb1, cb2, title_r)
