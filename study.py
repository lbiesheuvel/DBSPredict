
from pipeline import create_model, create_pipeline
from sklearn.model_selection import cross_val_score 
import optuna
from optuna.samplers import TPESampler


class DBSStudy():
    def __init__(self, X_train, y_train, random_state):
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        self.study = None
        self.performed_trial_params_and_results = []

    def objective(self, trial): 
        # Model selection
        model_name = trial.suggest_categorical('model', ['RandomForestRegressor', 'LinearRegression', 'Lasso'])
        # Hyperparameters for the RandomForestRegressor
        if model_name == 'RandomForestRegressor':
            trial.suggest_int('n_estimators', 20, 200),
            trial.suggest_int('max_depth', 3, 20),
            trial.suggest_int('min_samples_split', 2, 15),
            trial.suggest_int('min_samples_leaf', 1, 10)
        # Hyperparameters for Lasso
        elif model_name == 'Lasso':
            trial.suggest_float('alpha', 1e-2, 5, log=True)

        trial.suggest_categorical('remove_outliers', [True, False]) 
        model = create_model(trial.params)
        pipeline = create_pipeline(model, include_outlier_remover=trial.params['remove_outliers'], random_state=self.random_state)

        # check if this combination of parameters has already been tried. If yes, return score
        for params, score in self.performed_trial_params_and_results:
            if params == trial.params:
                print(f'Already tried this combination of parameters. Returning previous score ({score})')
                return score
        # Perform 5-fold cross-validation
        score = cross_val_score(pipeline, self.X_train, self.y_train, cv=5, scoring='neg_root_mean_squared_error').mean()
        self.performed_trial_params_and_results.append((trial.params, score))
        return score
    
    def run_study(self, n_trials):
        sampler = TPESampler(seed=self.random_state)  
        self.study = optuna.create_study(sampler=sampler, direction="maximize")
        self.study.optimize(self.objective, n_trials=n_trials)
        print(f'Total unique params tried: {len(self.performed_trial_params_and_results)}')

    def get_best_pipeline(self):
        best_model = create_model(self.study.best_params) 
        best_pipeline = create_pipeline(best_model, include_outlier_remover=self.study.best_params['remove_outliers'], random_state=self.random_state)
        best_pipeline.fit(self.X_train, self.y_train)
        return best_pipeline