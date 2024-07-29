from get_dataframe import get_dataframe
from sklearn.model_selection import ShuffleSplit


def get_data(candidate_features, key_features, outcome, return_X_y=True):
    """
    Function to preprocess and extract data for a study.

    Parameters:
    candidate_features (list): List of candidate features to include in the data.
    key_features (list): List of key features to include in the data.
    outcome (str): Name of the outcome variable.

    Returns:
    X (DataFrame): Preprocessed data with candidate features.
    y (Series): Outcome variable data.
    """
    data = get_dataframe()
    # get rid of records that didnt give permission to use data for research (1.0 means no permission, everthing else means permission)
    print(
        f'Out of {len(data)} records, {data["NO Permission data use for research"].value_counts()[1.0]} did not give permission to use data for research'
    )
    data = data[data["NO Permission data use for research"] != 1.0]

    data = data[["Participant Id"] + candidate_features + [outcome]]
    print(f"Amount of patients in data: {len(data)}")

    data = data.dropna(subset=key_features + [outcome])
    print(f"Amount after dropping rows that miss essential data: {len(data)}")
    # return X and y
    X = data[["Participant Id"] + candidate_features]
    y = data[outcome]
    if return_X_y:
        return X, y
    else:
        return data


def get_splits(X, y, test_size, n_splits, random_state):
    """
    Generate train-test splits of the given data.

    Parameters:
    X (pandas.DataFrame): The input features.
    y (pandas.Series): The target variable.
    test_size (float): The proportion of the dataset to include in the test split.
    n_splits (int): The number of times to split the data into train and test sets.
    random_state (int): The seed used by the random number generator.

    Returns:
    splits (list): A list of tuples containing the train-test splits.
                   Each tuple contains X_train, X_test, y_train, y_test.
    """
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    splits = []
    for train_index, test_index in ss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        splits.append((X_train, y_train, X_test, y_test))
    return splits
