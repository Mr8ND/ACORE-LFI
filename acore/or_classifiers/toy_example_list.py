from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression

classifier_dict = {
    'NN': KNeighborsClassifier(),
    'MLP': MLPClassifier(alpha=0, max_iter=25000),
    'QDA': QuadraticDiscriminantAnalysis()
}


classifier_dict_complete = {
    'NN': KNeighborsClassifier(),
    'MLP': MLPClassifier(alpha=0, max_iter=25000),
    'Gauss_Proc': GaussianProcessClassifier(RBF(1.0)),
    'XGBoost (d3, n500)': XGBClassifier(max_depth=3, n_estimators=500),
    'XGBoost (d3, n100)': XGBClassifier(max_depth=3, n_estimators=100),
    'XGBoost (d5, n500)': XGBClassifier(max_depth=3, n_estimators=500),
    'XGBoost \n (d10, n100)': XGBClassifier(max_depth=10, n_estimators=100),
    'QDA': QuadraticDiscriminantAnalysis(),
    'Log. Regr.': LogisticRegression(penalty='none', solver='saga', max_iter=10000)
}