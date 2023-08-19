import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA



if __name__ == '__main__':

    # set the seed so it is reproducible
    np.random.seed(13)

    # read the data
    # citation for the data: https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
    df = pd.read_csv('diabetes.txt', delim_whitespace=True)

    # make sure it read the file
    #print(df.head())

    # set the (column name) of the response variable for later use
    resp_var = 'Y'
    # explore_data(df, resp_var)

    # this will use the full dataset, later we can do train/test sets
    data = df.drop(resp_var, axis=1)
    labels = df[resp_var].copy()

    # it is important to scale the data
    std_scaler = StandardScaler().set_output(transform="pandas")
    std_scaler.fit(data)
    scaled_data = std_scaler.transform(data)

    # this will avoid scientific notation for decimals
    np.set_printoptions(suppress=True)

    pca = PCA()  # here we keep all components, later we will reduce
    pca.fit(scaled_data)
    print('number of components: ' + str(pca.n_components_))

    # variance explained by each component
    print(pca.explained_variance_ratio_)

    # how many components explain 95% of the variance?
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print('# of comp. explain 95% of variance:' + str(d))

    # print(pca.components_)
    print(pd.DataFrame(pca.components_, columns=scaled_data.columns, ))

    # visualize the components
    components = list(range(pca.components_.shape[0]))
    plt.bar(components, pca.explained_variance_ratio_)
    plt.xlabel("Component")
    plt.xticks(components)
    plt.ylabel("% of variance")
    plt.title('Components explained variance')
    plt.savefig('./images/explained-variance.png')
    plt.show()

    ############ try and relate the components back to original features ################
    # build PC's
    pca = PCA(n_components=0.95)  # this explains 95% of variance, could also do PCA(n_components=4)
    pca.fit(scaled_data)

    # build a df with the feature names as columns and the rows as principal components
    components_df = pd.DataFrame(pca.components_, columns=scaled_data.columns, )
    # this prints the column name of the maximum (absolute value) for each row
    important_features = components_df.abs().idxmax(axis=1)
    print(important_features)
    print(important_features.drop_duplicates())

    # get the column names as a list with no duplicate, for later use
    important_features_list = important_features.drop_duplicates().values.tolist()

    # # Now lets use it in a model
    # # pca = PCA(n_components=0.95) # Could also do it like this (explain 95%)
    # pca = PCA(n_components=4)
    # reduced_data = pca.fit_transform(scaled_data)
    # # create and fit linear regression model
    # lm = LinearRegression()
    # lm.fit(reduced_data, labels)
    # predict = lm.predict(reduced_data)

    ############### lets use all the features first ###########################333
    # # create and fit linear regression model
    lm = LinearRegression()
    lm.fit(data, labels)
    #print("Intercept: " + str(lm.intercept_))
    #print("Coefficients: " + str(lm.coef_))

    # make predictions
    pred = lm.predict(data)
    mse_score = mean_squared_error(labels, pred)
    r2_score_full = r2_score(labels, pred)

    # scores
    print("MSE full data: " + str(mse_score))
    print("R-Squared full data: " + str(r2_score_full))

    # cross validation
    scores = cross_val_score(lm, data, labels, cv=5, scoring='r2')
    print("Cross validation full data scores " + str(scores))
    avg_score = np.mean(np.absolute(scores))
    print("Average CV full data score: " + str(avg_score))

    ############### Now lets use the reduced features ###########################333
    # # create and fit linear regression model
    reduced_data = data.drop(columns=data.columns.difference(important_features_list))
    lm = LinearRegression()
    lm.fit(reduced_data, labels)
    #print("Intercept: " + str(lm.intercept_))
    #print("Coefficients: " + str(lm.coef_))

    # make predictions
    pred = lm.predict(reduced_data)
    mse_score = mean_squared_error(labels, pred)
    r2_score_reduced = r2_score(labels, pred)

    # scores
    print("MSE reduced data: " + str(mse_score))
    print("R-Squared reduced data: " + str(r2_score_reduced))

    # cross validation
    scores = cross_val_score(lm, reduced_data, labels, cv=5, scoring='r2')
    print("Cross validation reduced data scores " + str(scores))
    avg_score = np.mean(np.absolute(scores))
    print("Average CV reduced data score: " + str(avg_score))
    print("How much data was reduced: Dropped " + str(data.shape[1] - reduced_data.shape[1])
          + " features (" + str(int((1 - (reduced_data.shape[1] / data.shape[1])) * 100)) + "%)")


