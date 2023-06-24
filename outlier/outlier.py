import numpy as np
import pandas as pd
from outliers import smirnov_grubbs as grubbs
import matplotlib.pyplot as plt
from scipy.stats import shapiro


if __name__ == '__main__':

    # set the seed so it is reproducible
    np.random.seed(13)

    # read the data
    df = pd.read_csv('uscrime.txt', delim_whitespace=True)
    df = df['Crime'] # we are just checking the crime column for outliers

    # make sure it read the file
    #print(df.head())

    # view the data to get intuition and test for normality
    # histogram
    plt.clf()
    plt.hist(df)
    plt.title('Histogram Before')
    plt.savefig('./images/hist-before.png')
    plt.show()

    # box plot
    plt.clf()
    plt.boxplot(df)
    plt.title('Box Plot Before')
    plt.savefig('./images/box-before.png')
    plt.show()

    # shapiro test. p value greater than 0.05, the data is normal.
    # If it is below 0.05, the data will deviate from a normal distribution,
    # and you would reject the null hypothesis that your data is normally distributed
    print(shapiro(df))

    # grubbs test, p value greater than 0.05 then you cant reject the null hypothesis
    # if p value is less than 0.05 then you can say it is an outlier

    # the highest-crime city might be an outlier (p=0.079), and if we remove it,
    # the second-highest-crime city also appears to be an outlier (p=0.028).
    # so the alpha value had to be changed from 0.05 to 0.08
    max_indices = grubbs.max_test_indices(df, alpha=0.08)
    max_outliers = grubbs.max_test_outliers(df, alpha=0.08)
    min_indices = grubbs.min_test_indices(df, alpha=0.08)
    min_outliers = grubbs.min_test_outliers(df, alpha=0.08)

    print('Max outlier indices: ' + str(max_indices) + ', Max outlier values: ' + str(max_outliers))
    print('Min outlier indices: ' + str(min_indices) + ', Min outlier values: ' + str(min_outliers))

    # !!!!! be cautious of removing outliers, it may be better to
    # !!!!! investigate them more and find out why they are at the extremes

    # array with outliers removed
    outliers_removed = df.drop(max_indices, axis=0)

    # view the new data
    # histogram
    plt.clf()
    plt.hist(outliers_removed)
    plt.title('Histogram After')
    plt.savefig('./images/hist-after.png')
    plt.show()

    # box plot
    plt.clf()
    plt.boxplot(outliers_removed)
    plt.title('Box Plot After')
    plt.savefig('./images/box-after.png')
    plt.show()