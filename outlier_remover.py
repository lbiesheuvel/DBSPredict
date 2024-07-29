from sklearn.covariance import EllipticEnvelope
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, random_state):
        self.random_state = random_state

    def fit(self, X, y=None):
        outlier_detector = EllipticEnvelope(contamination=0.1, random_state=self.random_state)
        outlier_detector.fit(X)
        self.outliers = outlier_detector.predict(X)
        return self
    
    def transform(self, X, y):
        return X[self.outliers == 1], y[self.outliers == 1]
    
def outlier_removal(X, y):
    transformer = OutlierRemover(random_state=42)
    transformer.fit(X, y)
    return transformer.transform(X, y)