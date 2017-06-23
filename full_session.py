import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import os.path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from scipy import stats
import numpy as np
import statsmodels.api as sm

def main():

    ##### Populate list of lists of votes
    list_X = list()
    list_S = list()
    path = '/home/ubuntu/application/data/user_'
    i = 1
    fname = path + str(i) + '.csv'
    while os.path.isfile(fname): # For each user
       with open(fname, 'r') as g:
           reader = csv.reader(g)
           user_data = list(reader)[0]
           user_data.insert(0, .49)
           binary_data = list()
           summed_data = [0]
           for dat in range(1,(len(user_data)-1)):
               if float(user_data[dat]) >= float(user_data[dat-1]):
                   binary_data.append(1)
                   summed_data.append(summed_data[-1]+1)
               else:
                   binary_data.append(0)
                   summed_data.append(summed_data[-1]-1)
           list_X.append(binary_data)
           list_S.append(summed_data)
       i = i + 1
       fname = path + str(i) + '.csv'

    ##### Figure 1: visualize user behavior
    # For each user's data, plot a line
    for user in list_S:
        y_ = [float(j) for j in user]
        x_ = np.arange(0,len(y_))
        plt.plot(x_, y_, lw=3, alpha=0.3, color=(0,.4,.6))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    for y in range(-10, 15, 5):
        plt.plot(range(-1, 35), [y] * len(range(-1, 35)), "--", lw=0.5, color="black", alpha=0.3)
    plt.tick_params(axis="both", which="both", bottom="on", top="off",
                labelbottom="on", left="off", right="off", labelleft="off")
    plt.ylim(-12, 12)
    plt.xlim(-1, 35)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.savefig('/home/ubuntu/application/figure1a.png')
    plt.close()

    ## Additional figures highlighting particular users
    for u in range(0,len(list_S)):
        for user in list_S:
            y_ = [float(j) for j in user]
            x_ = np.arange(0,len(y_))
            plt.plot(x_, y_, lw=3, alpha=0.3, color=(0,.4,.6))
        y_ = list_S[u]
        x_ = np.arange(0,len(y_))
        plt.plot(x_, y_, lw=5, alpha=1, color="orange")
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        for y in range(-10, 15, 5):
            plt.plot(range(-1, 35), [y] * len(range(-1, 35)), "--", lw=0.5, color="black", alpha=0.3)
        plt.tick_params(axis="both", which="both", bottom="on", top="off",
                    labelbottom="on", left="off", right="off", labelleft="off")
        plt.ylim(-12, 12)
        plt.xlim(-1, 35)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        filename = '/home/ubuntu/application/data/figure' + str(u) + '.png'
        plt.savefig(filename)
        plt.close()

    ##### Model User Behavior
    # Linear regression: fb_sum = B0 + B1(fb_number)
    ps = list()
    rs = list()
    for user in list_S:
        y_ = [float(j) for j in user[:-1]]
        x_ = np.arange(0,len(y_))
        X_ = np.transpose(x_[np.newaxis])
        try:
            linear = sm.OLS(y_, X_)
            result = linear.fit(disp=0)
            coef = result.params
            pval = result.pvalues
            fail_ = False
        except:
            fail_ = True
        if coef > 0 and not fail_:
            ps.append(pval[0])
            rs.append(coef[0])
        else:
            ps.append(1)
            rs.append(0)
    print('\nLinear model p-values:')
    [print(p) for p in ps]

    ##### Figure 2: Visualize user's coefs
    # One set of connected points per user
    # X axis: num_clicks
    # Y axis: regression coefficient
    i = 0
    Xs = list()
    Ys = list()
    for user in list_S:
        Xs.append(len(np.transpose([float(j) for j in user[1:]])))
        Ys.append(rs[i])
        i = i + 1
    plt.plot(Xs, Ys, "o")
    plt.savefig('/home/ubuntu/application/figure2.png')
    plt.close()

    ##### Statistic 1
    # Did a significant proportion of users' regressions hit significance?
    # print(ps)
    usersSigOnOwn = sum(1 for p in ps if p <= .05)
    numUsers = len(list_S)
    print('\nStatistic 2:\nProportion of users significant:',usersSigOnOwn,'/',numUsers,'\n')
    usersSigOnOwn = sum(1 for p in ps if p == 1)
    print('\nStatistic 3:\nProportion of users -significant:',usersSigOnOwn,'/',numUsers,'\n')
    # In Octave:
    # pval = 1 - binocdf(usersSigOnOwn,numUsers,0.05)
    # pval = 1 - binocdf(14,31,0.05)
    # pval = 4.2595e-12

if __name__ == '__main__':
    main()
