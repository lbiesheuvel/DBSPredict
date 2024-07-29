from imblearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFECV
from imblearn import FunctionSampler
from outlier_remover import outlier_removal
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression


def create_model(params):
    model = params['model']
    if model == 'RandomForestRegressor':
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf']
        )
    elif model == 'LinearRegression':
        model = LinearRegression()
    elif model == 'Lasso':
        model = Lasso(
            alpha=params['alpha'], 
            max_iter=10000)
    else:
        raise ValueError(f'Unknown model name: {model}')
    return model
    

def create_pipeline(model, include_outlier_remover, random_state):
    pipeline = Pipeline(steps=[
        ('imputer', IterativeImputer(max_iter=1000, tol=1e-4, initial_strategy='median')),
        ('scaler', RobustScaler()),
        # We use a FunctionSampler, which is part of imblearn. 
        # We can use this to resample both X and y, which is needed for outlier removal.
        ('outlier_remover', FunctionSampler(func=outlier_removal)), 
        ('model', model)
    ])
    # Removed the RFECV step, because it shouldn't be part of the pipeline. Else, the imputed values are still dependent on the features that were not included by the RFECV.

    if not include_outlier_remover:
        # remove outlier remover from pipeline
        pipeline.steps.pop(2)
    return pipeline