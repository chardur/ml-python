import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
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
    plt.suptitle('Visualize the data')
    plt.savefig('./images/visual-data.png')
    plt.show()

    # Look for Correlations, R1 is response, other variables are predictors
    # correlation coeff ranges from –1 to 1. When it is close to 1, there is a strong positive correlation
    corr_matrix = df.corr().abs()
    print(corr_matrix[resp_var].sort_values(ascending=False))

    # visualize the correlation after selecting a few attributes (top 4_
    attributes = corr_matrix[resp_var].sort_values(ascending=False).iloc[0:4].index
    scatter_matrix(df[attributes], figsize=(12, 8))
    plt.suptitle('Correlation of top 4 features')
    plt.savefig('./images/correlation.png')
    plt.show()

    # Al has a somewhat spread out distribution, lets see what it looks like vs output
    df.plot(kind="scatter", x="Al", y="Type", grid=True, alpha=0.2)
    plt.title('Al vs Glass Type')
    plt.savefig('./images/one-vs-output.png')
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


def plot_knn_vertical(knn_clf, xmin, xmax, df_data, df_predict, df_actual):

    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('KNN accuracy comparison')
    #fig.tight_layout()
    fig.set_figheight(8)
    #fig.set_figwidth(15)

    # select first column as x and second column as y
    x = df_data.iloc[:, 0]
    y = df_data.iloc[:, 1]

    # scatter for predicted values
    axs[0].set_title('Predicted values')
    axs[0].scatter(x, y, c=df_predict, s=10, facecolors='#AAA', zorder=-1)

    # scatter for actual values
    axs[1].set_title('Actual values')
    axs[1].scatter(x, y, c=df_actual, s=10, facecolors='#AAA', zorder=-1)

    # first make a scatter for predicted values, then another scatter with a mask for errors
    classes = ['1', '2', '3', '4', '5', '6', '7']
    axs[2].set_title('Errors in prediction')
    scatter = axs[2].scatter(x, y, c=df_predict, s=10, facecolors='#AAA', zorder=-1)
    mask = df_predict != df_actual
    axs[2].scatter(x[mask==True], y[mask==True], s=50, facecolors='none', zorder=-1, edgecolors='r', label='Errors')

    # call legend first to display the currently defined labels
    axs[2].legend(loc='lower left')
    # create a new legend with some additional labels
    leg = Legend(axs[2], *scatter.legend_elements(), loc='lower right')
    axs[2].add_artist(leg)


if __name__ == '__main__':

    # set the seed so it is reproducible
    np.random.seed(13)

    # read the data
    df = pd.read_csv('glass.csv')
    #df = df[['Al', 'Mg', 'Type']]

    # make sure it read the file
    #print(df.head())

    # set the (column name) of the response variable for later use
    resp_var = 'Type'
    #explore_data(df, resp_var)

    # manually create train / test set using the shuffle_and_split_data function above, 20% test data
    # train_set, test_set = shuffle_and_split_data(df, 0.2)
    # print(test_set.head())

    # another way to create train / test set using sklearn
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=13)
    #print(test_set.head())

    # create a copy of the training & test set for future use with transformations
    # separate the predictors and the labels(response)
    glass_data_train = train_set.drop(resp_var, axis=1)
    glass_labels_train = train_set[resp_var].copy()
    glass_data_test = test_set.drop(resp_var, axis=1)
    glass_labels_test = test_set[resp_var].copy()

    # manually create classifier
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(glass_data_train, glass_labels_train)
    predict = knn_clf.predict(glass_data_test)
    print("KNN Accuracy (no scaling):", metrics.accuracy_score(glass_labels_test, predict))

    # classifier pipeline that does standardization scaling
    knn_clf_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
    knn_clf_pipe.fit(glass_data_train, glass_labels_train)
    predict = knn_clf_pipe.predict(glass_data_test)
    print("KNN Accuracy (standard scaling):", metrics.accuracy_score(glass_labels_test, predict))

    # classifier pipeline that does minmax scaling
    knn_clf_pipe = make_pipeline(MinMaxScaler(feature_range=(-1, 1)), KNeighborsClassifier())
    knn_clf_pipe.fit(glass_data_train, glass_labels_train)
    predict = knn_clf_pipe.predict(glass_data_test)
    print("KNN Accuracy (MinMax scaling):", metrics.accuracy_score(glass_labels_test, predict))

    # manually scaling the data
    std_scaler = StandardScaler().fit(glass_data_train)
    scaled_train = std_scaler.transform(glass_data_train)
    scaled_train = pd.DataFrame(scaled_train, index=glass_data_train.index, columns=glass_data_train.columns)
    scaled_test = std_scaler.transform(glass_data_test)
    scaled_test = pd.DataFrame(scaled_test, index=glass_data_test.index, columns=glass_data_test.columns)
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(scaled_train, glass_labels_train)
    predict = knn_clf.predict(scaled_test)
    print("KNN Accuracy (manual standard scaling):", metrics.accuracy_score(glass_labels_test, predict))

    # loop through k values 2-50 and find knn accuracy for each k
    knnScore = []
    knnValue = []
    bestK = 0
    bestScore = 0
    best_tie = []
    for i in range(2, 51):
        knn_clf = KNeighborsClassifier(n_neighbors=i)
        knn_clf.fit(glass_data_train, glass_labels_train)
        predict = knn_clf.predict(glass_data_test)
        score = metrics.accuracy_score(glass_labels_test, predict)
        knnScore.append(score)
        knnValue.append(i)
        if score > bestScore:
            bestK = i
            bestScore = score
            best_tie = []  # this clears out old tie scores that were lower
        if score == bestScore:
            best_tie.append(i)
    print('Best K value: ' + str(bestK) + ', Best score: ' + str(bestScore))
    print('Ties for best score: ' + str(best_tie))

    # graph the results for best k from above
    plt.plot(knnValue, knnScore, marker='o')
    plt.title("KNN K-Value vs Score")
    plt.xlabel("K Value")
    plt.ylabel("Score")
    # plt.xticks(np.arange(min(knnValue), max(knnValue) + 1, 5.0))
    plt.xticks(np.arange(0, max(knnValue) + 1, 5.0))
    plt.ylim(ymin=0)
    plt.ylim(ymax=1.0)
    plt.grid()

    # plot points for the ties for highest value
    for point in best_tie:
        plt.plot(point, knnScore[point-2], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="none")
    plt.show()

    knn_clf = KNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(glass_data_train, glass_labels_train)
    predict = knn_clf.predict(glass_data_test)
    plot_knn_vertical(knn_clf, 0, 250, glass_data_test, predict, glass_labels_test)
    plt.savefig('./images/vertical-accuracy.png')
    plt.show()

    # plt.scatter(glass_data_train.iloc[:, 0], glass_data_train.iloc[:, 1], c=glass_labels_train, s=10, facecolors='#AAA', zorder=-1)
    # plt.show()

