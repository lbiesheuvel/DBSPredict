from load_data import get_data, get_splits
from study import DBSStudy
from sklearn.metrics import mean_squared_error, r2_score
from plot_results import plot_results
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from imblearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
import shap


class Project:
    def __init__(self, candidate_features, project_outcome):
        if project_outcome == "Total UPDRS-III":
            self.baseline_feature_name = "Total UPDRS-III OFF score"
            self.key_features = [
                "Total UPDRS-III OFF score",
                "Total UPDRS-III ON score",
            ]
            self.outcome = "Postoperative total UPDRS-III OFF medication ON DBS score"
            self.outcome_abbreviation = "UPDRS-III Off Med On DBS"
            self.model_name = "Total UPDRS-III Model"
        elif project_outcome == "Bradykinesia + rigidity":
            self.baseline_feature_name = (
                "Total bradykinesia + rigidity OFF score (UPDRS-III subscore)"
            )
            self.key_features = [
                "Total bradykinesia + rigidity OFF score (UPDRS-III subscore)",
                "Total bradykinesia + rigidity ON score (UPDRS-III subscore)",
            ]
            self.outcome = "Postoperative total bradykinesia + rigidity OFF medication ON DBS score (UPDRS-III subscore)"
            self.outcome_abbreviation = "Rigidity Off Med On DBS"
            self.model_name = "Bradykinesia + rigidity Model"
        elif project_outcome == "Tremor":
            self.baseline_feature_name = "Total tremor OFF score (UPDRS-III subscore)"
            self.key_features = [
                "Total tremor OFF score (UPDRS-III subscore)",
                "Total tremor ON score (UPDRS-III subscore)",
            ]
            self.outcome = "Postoperative total tremor OFF medication ON DBS score (UPDRS-III subscore)"
            self.outcome_abbreviation = "Tremor Off Med On DBS"
            self.model_name = "Tremor Model"

        elif project_outcome == "Axial":
            self.baseline_feature_name = "Axial OFF score (UPDRS-III subscore)"
            self.key_features = [
                "Axial OFF score (UPDRS-III subscore)",
                "Axial ON score (UPDRS-III subscore)",
            ]
            self.outcome = (
                "Postoperative axial OFF medication ON DBS score (UPDRS-III subscore)"
            )
            self.outcome_abbreviation = "Axial Off Med On DBS"
            self.model_name = "Axial Model"
        else:
            raise ValueError(f"Unknown project outcome: {project_outcome}")

        self.CANDIDATE_FEATURES = candidate_features
        self.RND = 42

        self.X, self.y = get_data(
            candidate_features=self.CANDIDATE_FEATURES,
            key_features=self.key_features,
            outcome=self.outcome,
        )

        # get indices of X and save those
        self.included_indices = self.X.index
        splits = get_splits(
            self.X, self.y, test_size=0.2, n_splits=1, random_state=self.RND
        )  # returns list of n_splits tuples (X_train, X_test, y_train, y_test)
        self.X_train, self.y_train, self.X_test, self.y_test = splits[0]
        self.X_test_copy = (
            self.X_test.copy()
        )  # Copy that includes participant id, used for printing outliers
        # remove participant id from X_train and X_test
        self.X_train = self.X_train.drop(columns=["Participant Id"])
        self.X_test = self.X_test.drop(columns=["Participant Id"])
        # Linear regression for feature selection
        feature_selection_model = LinearRegression()
        feature_selector = Pipeline(
            steps=[
                (
                    "imputer",
                    IterativeImputer(
                        max_iter=1000, tol=1e-4, initial_strategy="median"
                    ),
                ),
                ("scaler", RobustScaler()),
                (
                    "feature_selection",
                    RFECV(
                        estimator=feature_selection_model,
                        step=1,
                        cv=KFold(n_splits=5, shuffle=True, random_state=self.RND),
                        scoring="neg_mean_squared_error",
                    ),
                ),
            ]
        )

        feature_selector.fit(self.X_train, self.y_train)
        columns_to_keep = feature_selector.named_steps[
            "feature_selection"
        ].get_support()
        self.X_train = self.X_train.loc[:, columns_to_keep]
        self.X_test = self.X_test.loc[:, columns_to_keep]

        selected_features = feature_selector.named_steps[
            "feature_selection"
        ].get_support()
        print("Selected features:")

        self.list_included_features = [
            feature
            for i, feature in enumerate(self.CANDIDATE_FEATURES)
            if selected_features[i]
        ]
        for feature in self.list_included_features:
            print(f" - {feature}")
        self.num_training_records = len(self.X_train)
        self.num_testing_records = len(self.X_test)
        self.y_pred = None
        self.best_pipeline = None
        self.performance = None
        self.lower_bound = None
        self.upper_bound = None

        self.rsquaredperformance = None
        self.rsquaredlower_bound = None
        self.rsquaredupper_bound = None

    def train_and_tune_model(self, n_trials):
        study = DBSStudy(self.X_train, self.y_train, random_state=self.RND)
        study.run_study(n_trials=n_trials)
        print(f"Best params: {study.study.best_params}")
        self.best_model_type = study.study.best_params["model"]
        self.best_pipeline = study.get_best_pipeline()

    def evaluate(self):
        if self.best_pipeline is None:
            print("Model not trained yet")
            return
        y_pred = self.best_pipeline.predict(self.X_test)
        y_pred = y_pred.round()

        # Evaluate the model
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        # Print the scores
        print(f"RMSE: {rmse:.2f}")

        r2 = r2_score(self.y_test, y_pred)
        print(f"R^2: {r2:.2f}")

        # We determine the biggest outliers
        biggest_outliers = np.argsort(-np.abs(self.y_test - y_pred))[:5]
        print("Biggest outliers:")
        for i in biggest_outliers:
            participant_id = self.X_test_copy.iloc[i]["Participant Id"]
            print(
                f"Participant {participant_id}: True: {self.y_test.iloc[i]}, Predicted: {y_pred[i]}"
            )

        plot_results(
            pipeline=self.best_pipeline,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            y_pred=y_pred,
            baseline_feature_name=self.baseline_feature_name,
            outcome_name=self.outcome_abbreviation,
        )

        self.performance = rmse
        self.rsquaredperformance = r2

    def calculate_confidence_interval(self, n_bootstraps):
        if self.best_pipeline is None:
            print("Model not trained yet")
            return
        lower_percentile = 2.5
        upper_percentile = 97.5
        # Create n_bootstraps bootstrap samples of the testing set
        testing_samples = []  # list of tuples (X_test, y_test)
        # make numpy determinisitc
        np.random.seed(self.RND)
        for i in range(n_bootstraps):
            # Resample with replacement, meaning that some samples will be repeated.
            # This is what we want, because it simulates the situation where we have a dataset of n samples, and we want to create a new dataset of n samples.
            # The mean should be about the same, but the samples will be different.

            new_test_sample_indices = np.random.choice(
                self.X_test.index, size=len(self.X_test), replace=True
            )
            new_X_test = self.X_test.loc[new_test_sample_indices]
            new_y_test = self.y_test.loc[new_test_sample_indices]
            testing_samples.append((new_X_test, new_y_test))

        # Calculate the rmse score for each bootstrap sample
        rmse_scores = []
        for X_test, y_test in testing_samples:
            y_pred = self.best_pipeline.predict(X_test).round()
            rmse_scores.append(mean_squared_error(y_test, y_pred, squared=False))

        # Calculate the confidence interval
        lower_rmse = np.percentile(rmse_scores, lower_percentile)
        upper_rmse = np.percentile(rmse_scores, upper_percentile)
        print(f"Confidence interval RMSE: [{lower_rmse:.2f}, {upper_rmse:.2f}]")

        rsquared_scores = []
        for X_test, y_test in testing_samples:
            y_pred = self.best_pipeline.predict(X_test).round()
            rsquared_scores.append(r2_score(y_test, y_pred))

        # Calculate the confidence interval
        lower_rsquared = np.percentile(rsquared_scores, lower_percentile)
        upper_rsquared = np.percentile(rsquared_scores, upper_percentile)
        print(f"Confidence interval R^2: [{lower_rsquared:.2f}, {upper_rsquared:.2f}]")

        # Plot results
        plt.hist(rmse_scores, bins=20)
        plt.xlabel("RMSE")
        plt.ylabel("Frequency")
        plt.title("RMSE distribution")
        plt.savefig(
            f"paper/img/bootstrap_results_rmse_{self.model_name}.pdf", format="pdf"
        )
        plt.show()

        self.lower_bound = lower_rmse
        self.upper_bound = upper_rmse

        self.rsquaredlower_bound = lower_rsquared
        self.rsquaredupper_bound = upper_rsquared

    def perform_shap_analysis(self):
        # set overall font size for ALL plots in matplotlib

        # Shows a summary plot of the SHAP values for each feature on the test set
        if self.best_pipeline is None:
            print("Model not trained yet")
            return
        explainer = shap.Explainer(self.best_pipeline.predict, self.X_train)
        shap_values = explainer(self.X_test)
        # get number of features to show in beeswarm
        n_features = min(20, len(self.X_test.columns))
        y_size = round(max(1 * n_features, 1.5))
        longest_column_size = max([len(str(x)) for x in self.X_test.columns])
        x_size = round(longest_column_size * 0.12) + 6

        shap.plots.beeswarm(shap_values, plot_size=(x_size, round(y_size)), show=False)
        plt.tight_layout()
        plt.savefig(f"paper/img/shap_beeswarm_{self.model_name}.pdf", format="pdf")
        plt.show()
        # plt.tight_layout()

        # summary plot
        shap.summary_plot(
            shap_values,
            self.X_test,
            plot_type="bar",
            plot_size=(x_size, round(y_size)),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(f"paper/img/shap_summary_{self.model_name}.pdf", format="pdf")
        plt.show()
        # plt.tight_layout()
        # dependence plots for top 4 features
        sorted_features = np.argsort(-np.abs(shap_values.values).mean(0))
        top_features = sorted_features[:4]
        if len(top_features) < 4:
            _, axes = plt.subplots(
                1, len(top_features), figsize=(len(top_features) * 8, 8)
            )
        else:
            _, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes = axes.flatten()
        for i, feature in enumerate(top_features):
            shap.dependence_plot(
                feature, shap_values.values, self.X_test, ax=axes[i], show=False
            )
        plt.tight_layout()
        plt.show()
