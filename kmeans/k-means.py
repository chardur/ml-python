import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
from matplotlib.legend import Legend
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


# citation Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow (3rd edition)
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# a function that runs some basic data exploration, this is optional
def explore_data(df, resp_var, sequence=1):
    # Quick description of data, make sure the non-null values match, otherwise missing data
    print(df.info())

    # count how many values in a specific category, useful for strings that repeat
    print(df[resp_var].value_counts())

    # a summary of the data
    print(df.describe())

    # visualize the data to get an initial feel for it
    df.hist(bins=50, figsize=(12, 8))
    plt.suptitle('Visualize the data')
    plt.savefig('./images/visual-data' + str(sequence) + '.png')
    plt.show()

    # Look for Correlations, R1 is response, other variables are predictors
    # correlation coeff ranges from –1 to 1. When it is close to 1, there is a strong positive correlation
    corr_matrix = df.corr().abs()
    print(corr_matrix[resp_var].sort_values(ascending=False))

    # visualize the correlation after selecting a few attributes (top 4_
    attributes = corr_matrix[resp_var].sort_values(ascending=False).iloc[0:4].index
    scatter_matrix(df[attributes], figsize=(12, 8))
    plt.suptitle('Correlation of top 4 features')
    plt.savefig('./images/correlation' + str(sequence) + '.png')
    plt.show()

    # Al has a somewhat spread out distribution, lets see what it looks like vs output
    df.plot(kind="scatter", x="Petal.Width", y=resp_var, grid=True, alpha=0.2)
    plt.title('Petal Width vs Species')
    plt.savefig('./images/one-vs-output' + str(sequence) + '.png')
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


def plot_kmeans_vertical(kmeans_clf, xmin, xmax, df_data, df_predict, df_actual, categories):


    # # colors = {'setosa': 'blue', 'versicolor': 'green', 'virginica': 'yellow'}
    # classes = ['setosa', 'versicolor', 'virginica']
    # # colors = {0: 'blue', 1: 'orange', 2: 'green'}  # 0=setosa, 1=versicolor, 2=virginica
    # scatter = plt.scatter(df['Petal.Length'], df['Petal.Width'], c=df['Species'])
    # plt.legend(handles=scatter.legend_elements()[0], labels=classes, title='Species')
    # plt.show()

    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('Kmeans accuracy comparison')
    #fig.tight_layout()
    fig.set_figheight(8)
    #fig.set_figwidth(15)

    # select third column as x and fourth column as y
    x = df_data.iloc[:, 2]
    y = df_data.iloc[:, 3]

    # scatter for predicted values
    axs[0].set_title('Predicted values')
    axs[0].scatter(x, y, c=df_predict, s=10, facecolors='#AAA', zorder=-1)

    # scatter for actual values
    axs[1].set_title('Actual values')
    axs[1].scatter(x, y, c=df_actual, s=10, facecolors='#AAA', zorder=-1)

    # first make a scatter for predicted values, then another scatter with a mask for errors
    #classes = ['1', '2', '3', '4', '5', '6', '7']
    classes = categories
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
    df = pd.read_csv('iris.txt', delim_whitespace=True)
    #df = df[['Al', 'Mg', 'Type']]

    # make sure it read the file
    #print(df.head())

    # encode text response var to numbers
    # note: if this was not the resp var you should do one hot encoding instead
    ordinal_encoder = OrdinalEncoder()
    df['Species'] = ordinal_encoder.fit_transform(df[['Species']])
    cats = ordinal_encoder.categories_[0].tolist()

    # set the (column name) of the response variable for later use
    resp_var = 'Species'
    #explore_data(df, resp_var)


    # manually create train / test set using the shuffle_and_split_data function above, 20% test data
    # train_set, test_set = shuffle_and_split_data(df, 0.2)

    # another way to create train / test set using sklearn
    # train_set, test_set = train_test_split(df, test_size=0.2, random_state=13)

    # create a copy of the training & test set for future use with transformations
    # separate the predictors and the labels(response)
    # data_train = train_set.drop(resp_var, axis=1)
    # labels_train = train_set[resp_var].copy()
    # data_test = test_set.drop(resp_var, axis=1)
    # labels_test = test_set[resp_var].copy()

    # let's use this simple way to create train / test set
    # note: df.iloc[:, :-1] only works if the response variable is the last column in the df
    data_train, data_test, labels_train, labels_test = train_test_split(
        df.iloc[:, :-1], df[resp_var], test_size=0.2)

    # inertia_: Sum of squared distances of samples to their closest cluster center
    # elbow method: loop over k values and plot, look for a bend and use it as optimal k value
    distances = []  # this is a list of inertia from sklearn kmeans object
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=13, n_init=20)
        kmeans.fit(data_train)
        distances.append(kmeans.inertia_)
    plt.plot(range(2, 8), distances, marker='o')
    plt.title('Elbow curve')
    plt.xlabel('K value')
    plt.ylabel('Sum of squared distances (inertia)')
    plt.savefig('./images/elbow.png')
    plt.show()

    # elbow bends at k=3, k=4, k=5, see which is more accurate
    # we happen to know the species, normally we wouldn't know this for unsupervised
    plot_k = [3, 4, 5]
    plot_train_score = []
    plot_test_score = []
    for k in range(3, 6):
        kmeans = KMeans(n_clusters=k, random_state=13, n_init=20)
        kmeans.fit(data_train)
        score_train = metrics.accuracy_score(labels_train, kmeans.predict(data_train))
        score_test = metrics.accuracy_score(labels_test, kmeans.predict(data_test))
        plot_train_score.append(score_train)
        plot_test_score.append(score_test)


    plt.plot(plot_k, plot_train_score, color='orange', marker='o', label='Train')
    plt.plot(plot_k, plot_test_score, color='blue', marker='o', label='Test')
    plt.title('K-Value vs Accuracy')
    plt.xlabel('K value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./images/accuracy.png')
    plt.show()


    k = 3
    kmeans = KMeans(n_clusters=k, random_state=13, n_init=20)
    kmeans.fit(data_train)
    score_train = metrics.accuracy_score(labels_train, kmeans.predict(data_train))
    pred_test = kmeans.predict(data_test)  # save this for later use with graph
    score_test = metrics.accuracy_score(labels_test, kmeans.predict(data_test))
    print(score_train)
    print(score_test)

    # try it with scaled data
    # manually scaling the data
    std_scaler = StandardScaler().fit(data_train)
    scaled_train = std_scaler.transform(data_train)
    scaled_test = std_scaler.transform(data_test)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=13, n_init=20)
    kmeans.fit(scaled_train)
    score_train = metrics.accuracy_score(labels_train, kmeans.predict(scaled_train))
    pred_test = kmeans.predict(scaled_test)  # save this for later use with graph
    score_test = metrics.accuracy_score(labels_test, kmeans.predict(scaled_test))
    print(score_train)
    print(score_test)

    # # try it with scaled data
    # # manually scaling the data
    # mmscaler = MinMaxScaler(feature_range=(-1, 1)).fit(data_train)
    # scaled_train = mmscaler.transform(data_train)
    # scaled_train = pd.DataFrame(scaled_train, index=data_train.index, columns=data_train.columns)
    # scaled_test = mmscaler.transform(data_test)
    # scaled_test = pd.DataFrame(scaled_test, index=data_test.index, columns=data_test.columns)
    # k = 3
    # kmeans = KMeans(n_clusters=k, random_state=13, n_init=20)
    # pred = kmeans.fit_predict(scaled_train)
    # score_train = metrics.accuracy_score(labels_train, kmeans.predict(scaled_train))
    # pred_test = kmeans.predict(scaled_test)  # save this for later use with graph
    # score_test = metrics.accuracy_score(labels_test, kmeans.predict(scaled_test))
    # print(score_train)
    # print(score_test)
    #
    # # try using a pipeline to scale and fit
    # kmeans_clf_pipe = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=13, n_init=20))
    # kmeans_clf_pipe.fit_predict(data_train, labels_train)
    # predict = kmeans_clf_pipe.predict(data_test)
    # print(metrics.accuracy_score(labels_test, predict))

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=13, n_init=20)
    kmeans.fit(data_train)
    pred_test = kmeans.predict(data_test)  # save this for later use with graph
    plot_kmeans_vertical(kmeans, 0, 250, data_test, pred_test, labels_test, cats)
    plt.savefig('./images/vertical-accuracy.png')
    plt.show()

