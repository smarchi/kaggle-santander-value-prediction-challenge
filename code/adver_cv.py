def create_adv_train(train, test):

    """
    Parameters:
    train: train set. Type: pandas dataframe
    test: test set. Type: pandas dataframe
    """

    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import model_selection as CV
    from sklearn.metrics import accuracy_score

    #add 'is_test column. train = 0, test = 1
    train['is_test'] = 0.
    test['is_test'] = 1.

    #concatenate and shuffle
    all_data = pd.concat((train,test))
    all_data = all_data.iloc[np.random.permutation(len(all_data))]

    #train random forest
    x = all_data.drop('is_test', axis = 1)
    y = all_data['is_test']

    cv = CV.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

    predictions_prob = np.zeros(len(y))

    for f, (train_i, test_i) in enumerate(cv.split(x, y)):

        print("Analyzing Fold {}".format(f+1))

        x_train = np.array(x.iloc[train_i])
        y_train = np.array(y.iloc[train_i])
        x_test = np.array(x.iloc[test_i])
        y_test = np.array(y.iloc[test_i])

        clf = RandomForestClassifier(n_estimators = 100, random_state = 123)
        clf.fit(np.array(x_train), np.array(y_train))

        p = clf.predict_proba(x_test)[:,1]

        predictions_prob[test_i] = p

        #accuracy
        ac = accuracy_score(y_test, clf.predict(x_test))
        print("Accuracy = {:.2%}\n".format(ac))

    #return train dataset with probability of being test column and ordered
    all_data['p'] = predictions_prob
    train_p = all_data.loc[all_data['is_test'] == 0]
    train_p.sort_index(inplace = True)
    train_p.drop('is_test', inplace=True, axis = 1)
    return train_p
