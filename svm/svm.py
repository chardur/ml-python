import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.svm import SVC
from matplotlib.legend import Legend


# citation Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow (3rd edition)
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# a function that runs some basic data exploration, this is optional
def explore_data(df, resp_var):
    # Quick description of data, make sure the non-null values match, otherwise missing data
    print(df.info())

    # count how many values in a specific category, useful for strings that repeat
    print(df[resp_var].value_counts())

    # a summary of the data
    print(df.describe())

    # visualize the data to get an initial feel for it
    df.hist(bins=50, figsize=(12, 8))
    plt.show()

    # Look for Correlations, R1 is response, other variables are predictors
    # correlation coeff ranges from –1 to 1. When it is close to 1, there is a strong positive correlation
    corr_matrix = df.corr().abs()
    print(corr_matrix[resp_var].sort_values(ascending=False))

    # visualize the correlation after selecting a few attributes (top 4_
    attributes = corr_matrix[resp_var].sort_values(ascending=False).iloc[0:4].index
    scatter_matrix(df[attributes], figsize=(12, 8))
    plt.show()

    # oldpeak has a somewhat spread out distribution, lets see what it looks like vs output
    df.plot(kind="scatter", x="oldpeak", y="output", grid=True, alpha=0.2)
    plt.show()


# clean the data in case of missing features, uses imputation to replace, only does numbers!
# option can be: median, mean, most_frequent
def clean_data(df, option="median"):
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy=option)
    df_num = df.select_dtypes(include=[np.number])
    imputer.fit(df_num)
    temp = imputer.transform(df_num)
    df_clean = pd.DataFrame(temp, columns=df_num.columns, index=df_num.index)
    return df_clean


def plot_svc(svm_clf, xmin, xmax, df_data, df_predict, df_actual):
    x0 = np.linspace(xmin, xmax, 200)
    b = svm_clf.intercept_[0]
    w = svm_clf.coef_[0]
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]

    margin = 1/w[1]
    margin_up = decision_boundary + margin
    margin_down = decision_boundary - margin
    svs = svm_clf.support_vectors_

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('SVM boundary and accuracy comparison')

    axs[0, 0].set_title('Actual values')
    axs[0, 0].plot(x0, decision_boundary, "k-", linewidth=2, zorder=-2)
    axs[0, 0].plot(x0, margin_up, "k--", linewidth=2, zorder=-2)
    axs[0, 0].plot(x0, margin_down, "k--", linewidth=2, zorder=-2)

    axs[0, 1].set_title('Predicted values')
    axs[0, 1].plot(x0, decision_boundary, "k-", linewidth=2, zorder=-2)
    axs[0, 1].plot(x0, margin_up, "k--", linewidth=2, zorder=-2)
    axs[0, 1].plot(x0, margin_down, "k--", linewidth=2, zorder=-2)

    axs[1, 0].set_title('Errors in prediction')
    axs[1, 0].plot(x0, decision_boundary, "k-", linewidth=2, zorder=-2)
    axs[1, 0].plot(x0, margin_up, "k--", linewidth=2, zorder=-2)
    axs[1, 0].plot(x0, margin_down, "k--", linewidth=2, zorder=-2)

    # this is just to get the axis on the bottom right box
    axs[1, 1].plot(x0, decision_boundary, "k-", linewidth=2, zorder=-2)
    axs[1, 1].plot(x0, margin_up, "k--", linewidth=2, zorder=-2)
    axs[1, 1].plot(x0, margin_down, "k--", linewidth=2, zorder=-2)

    x = df_data.iloc[:, 0]
    y = df_data.iloc[:, 1]

    axs[0, 0].scatter(x, y, c=df_actual, s=10, facecolors='#AAA', zorder=-1)
    axs[0, 1].scatter(x, y, c=df_predict, s=10, facecolors='#AAA', zorder=-1)

    axs[1, 0].scatter(x, y, c=df_predict, s=10, facecolors='#AAA', zorder=-1)
    mask = df_predict != df_actual
    axs[1, 0].scatter(x[mask==True], y[mask==True], s=50, facecolors='none', zorder=-1, edgecolors='r')

    for ax in fig.get_axes():
        ax.label_outer()


# citation Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow (3rd edition)
def plot_svc_vertical(svm_clf, xmin, xmax, df_data, df_predict, df_actual):
    x0 = np.linspace(xmin, xmax, 200)
    b = svm_clf.intercept_[0]
    w = svm_clf.coef_[0]
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]

    margin = 1/w[1]
    margin_up = decision_boundary + margin
    margin_down = decision_boundary - margin
    svs = svm_clf.support_vectors_

    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('SVM boundary and accuracy comparison')
    #fig.tight_layout()
    fig.set_figheight(8)
    #fig.set_figwidth(15)

    # this makes the boundary and margin lines for the 3 subplots
    axs[0].set_title('Predicted values')
    axs[0].plot(x0, decision_boundary, "k-", linewidth=2, zorder=-2)
    axs[0].plot(x0, margin_up, "k--", linewidth=2, zorder=-2)
    axs[0].plot(x0, margin_down, "k--", linewidth=2, zorder=-2)

    axs[1].set_title('Actual values')
    axs[1].plot(x0, decision_boundary, "k-", linewidth=2, zorder=-2)
    axs[1].plot(x0, margin_up, "k--", linewidth=2, zorder=-2)
    axs[1].plot(x0, margin_down, "k--", linewidth=2, zorder=-2)

    axs[2].set_title('Errors in prediction')
    axs[2].plot(x0, decision_boundary, "k-", linewidth=2, zorder=-2, label='Boundary')
    axs[2].plot(x0, margin_up, "k--", linewidth=2, zorder=-2, label='Margin')
    axs[2].plot(x0, margin_down, "k--", linewidth=2, zorder=-2)

    # select first column as x and second column as y
    x = df_data.iloc[:, 0]
    y = df_data.iloc[:, 1]

    # scatter for predicted values
    axs[0].scatter(x, y, c=df_predict, s=10, facecolors='#AAA', zorder=-1)
    # scatter for actual values
    axs[1].scatter(x, y, c=df_actual, s=10, facecolors='#AAA', zorder=-1)

    # first make a scatter for predicted values, then another scatter with a mask for errors
    classes = ['At Risk', 'Low Risk']
    scatter = axs[2].scatter(x, y, c=df_predict, s=10, facecolors='#AAA', zorder=-1)
    mask = df_predict != df_actual
    axs[2].scatter(x[mask==True], y[mask==True], s=50, facecolors='none', zorder=-1, edgecolors='r', label='Errors')

    # call legend first to display the currently defined labels
    axs[2].legend(loc='lower left')
    # create a new legend with some additional labels
    leg = Legend(axs[2], scatter.legend_elements()[0], classes, loc='lower right')
    axs[2].add_artist(leg)


if __name__ == '__main__':

    # set the seed so it is reproducible
    np.random.seed(13)

    # read the data
    df = pd.read_csv('heart.csv')
    #df = df[['chol', 'age', 'output']]
    df = df[['thalachh', 'oldpeak', 'output']]

    # make sure it read the file
    # print(df.head())

    # set the (column name) of the response variable for later use
    resp_var = 'output'
    #explore_data(df, resp_var)

    # manually create train / test set using the shuffle_and_split_data function above, 20% test data
    # train_set, test_set = shuffle_and_split_data(df, 0.2)
    # print(test_set.head())

    # another way to create train / test set using sklearn
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=13)
    #print(test_set.head())

    # create a copy of the training & test set for future use with transformations
    # separate the predictors and the labels(response)
    heart_data_train = train_set.drop(resp_var, axis=1)
    heart_labels_train = train_set[resp_var].copy()
    heart_data_test = test_set.drop(resp_var, axis=1)
    heart_labels_test = test_set[resp_var].copy()

    # manually scaling the data
    # std_scaler = StandardScaler()
    # heart_data_std_scaled = std_scaler.fit_transform(heart_data.values)
    # heart_data_std_scaled_df = pd.DataFrame(heart_data_std_scaled, index=heart_data.index, columns=heart_data.columns)
    # plt.show()

    # linear classifier pipeline that does standardization scaling
    svm_clf = make_pipeline(StandardScaler(), LinearSVC(C=1, random_state=13))
    svm_clf.fit(heart_data_train, heart_labels_train)
    predict = svm_clf.predict(heart_data_test)
    print("Linear Accuracy (standard scaling):", metrics.accuracy_score(heart_labels_test, predict))
    # print the equation of the sv
    # print(svm_clf.named_steps['linearsvc'].coef_)
    # print(svm_clf.named_steps['linearsvc'].intercept_)

    # testing SVC vs previous linearSVC method
    svc_svm = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1, random_state=13))
    svc_svm.fit(heart_data_train, heart_labels_train)
    predict = svc_svm.predict(heart_data_test)
    print("SVC Linear Accuracy (standard scaling):", metrics.accuracy_score(heart_labels_test, predict))

    # how does no scaling compare? this example is not using a pipeline
    no_scale_svm = SVC(kernel="linear", C=1)
    no_scale_svm.fit(heart_data_train, heart_labels_train)
    predict = no_scale_svm.predict(heart_data_test)
    print("SVC Linear Accuracy (no scaling):", metrics.accuracy_score(heart_labels_test, predict))

    #  Radial Basis Function RBF classifier pipeline that does standardization scaling
    rbf_kernel_svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=1, C=1))
    rbf_kernel_svm_clf.fit(heart_data_train, heart_labels_train)
    predict = rbf_kernel_svm_clf.predict(heart_data_test)
    print("RBF Accuracy (standard scaling):", metrics.accuracy_score(heart_labels_test, predict))

    # linear classifier pipeline that does minmax scaling
    svm_clf = make_pipeline(MinMaxScaler(feature_range=(-1, 1)), LinearSVC(C=1, random_state=13))
    svm_clf.fit(heart_data_train, heart_labels_train)
    predict = svm_clf.predict(heart_data_test)
    print("Linear Accuracy (MinMax scaling):", metrics.accuracy_score(heart_labels_test, predict))

    # RBF classifier pipeline that does minmax scaling
    rbf_kernel_svm_clf = make_pipeline(MinMaxScaler(feature_range=(-1, 1)), SVC(kernel="rbf", gamma=1, C=1))
    rbf_kernel_svm_clf.fit(heart_data_train, heart_labels_train)
    predict = rbf_kernel_svm_clf.predict(heart_data_test)
    print("RBF Accuracy (MinMax scaling):", metrics.accuracy_score(heart_labels_test, predict))

    # experiment to find the best C value
    # i = 0.001
    # bestC = 0
    # bestScore = 0
    # scoreVector = []
    # cVector = []
    # while (i < 1000):
    #     svc_svm = make_pipeline(StandardScaler(), SVC(kernel='linear', C=i, random_state=13))
    #     svc_svm.fit(heart_data_train, heart_labels_train)
    #     predict = svc_svm.predict(heart_data_test)
    #     score = metrics.accuracy_score(heart_labels_test, predict)
    #     scoreVector.append(score)
    #     cVector.append(i)
    #     if (score > bestScore):
    #         bestC = i
    #         bestScore = score
    #     i = i * 10
    #
    # plt.plot(cVector, scoreVector)
    # plt.xlabel("C value")
    # plt.ylabel("Score")
    # plt.show()

    # c value didnt matter much

    # print(svm_clf.named_steps['linearsvc'].coef_)
    # print(svm_clf.named_steps['linearsvc'].intercept_)

    # plot_svc(no_scale_svm, 0, 500, heart_data_test, predict, heart_labels_test)
    # plt.show()

    plot_svc_vertical(no_scale_svm, 0, 250, heart_data_test, predict, heart_labels_test)
    plt.show()
