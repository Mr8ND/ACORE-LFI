from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from or_classifiers.dnn_classifiers import OddsNet

# classifier_dict = {
#     'NN': KNeighborsClassifier(),
#     'MLP': MLPClassifier(alpha=0, max_iter=25000),
#     'QDA': QuadraticDiscriminantAnalysis()
# }

classifier_dict = {
    'pytorch_mlp': OddsNet(direct_odds=False, batch_size=64, learning_rate=1e-6),
    'pytorch_mlp_direct': OddsNet(direct_odds=True, batch_size=64, learning_rate=1e-5),
    'MLP': MLPClassifier(alpha=0, max_iter=25000)
}

classifier_dict_multid = {
    'NN': KNeighborsClassifier(),
    'MLP': MLPClassifier(alpha=0, max_iter=25000),
    'XGBoost (d3, n500)': XGBClassifier(max_depth=3, n_estimators=500)
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