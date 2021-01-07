from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from or_classifiers.dnn_classifiers import OddsNet

classifier_dict = {
    'NN': KNeighborsClassifier(),
    'MLP': MLPClassifier(alpha=0, max_iter=25000),
    'QDA': QuadraticDiscriminantAnalysis()
}

classifier_inferno_dict = {
    'XGBoost \n (d3, n100)': XGBClassifier(n_estimators=100)
}

classifier_pvalue_dict = {
    'MLP': MLPClassifier(alpha=0, max_iter=25000),
    'MLP2': MLPClassifier((64, 32, 32), activation='relu', alpha=0, max_iter=25000),
    'XGBoost (d3, n500)': XGBClassifier(max_depth=3, n_estimators=500),
    'Log. Regr.': LogisticRegression(penalty='none', solver='saga', max_iter=10000)
}

classifier_dict_mlpcomp = {
    'pytorch_mlp_orloss': OddsNet(loss_function='or_loss', batch_size=64,
                                  learning_rate=1e-6, n_epochs=50000),
    'pytorch_mlp_ordirectloss': OddsNet(loss_function='direct_odds', batch_size=64,
                                        learning_rate=1e-5, n_epochs=50000),
    # 'pytorch_mlp_kl_or': OddsNet(loss_function='kl_or', batch_size=64, verbose=True, epoch_check=250,
    #                                learning_rate=1e-5, n_epochs=50000),
    'MLP': MLPClassifier(alpha=0, max_iter=25000)
}

classifier_dict_multid = {
    'NN': KNeighborsClassifier(),
    'MLP': MLPClassifier(alpha=0, max_iter=25000),
    'MLP2': MLPClassifier((64, 32, 32), activation='relu', alpha=0, max_iter=25000)
}

classifier_dict_multid_power = {
    'MLP': MLPClassifier(alpha=0, max_iter=25000)
}

classifier_dict_complete = {
    'NN': KNeighborsClassifier(),
    'MLP': MLPClassifier(alpha=0, max_iter=25000),
    'Gauss_Proc': GaussianProcessClassifier(RBF(1.0)),
    'XGBoost (d3, n500)': XGBClassifier(max_depth=3, n_estimators=500),
    'XGBoost (d3, n100)': XGBClassifier(max_depth=3, n_estimators=100),
    'XGBoost (d5, n500)': XGBClassifier(max_depth=5, n_estimators=500),
    'XGBoost \n (d10, n100)': XGBClassifier(max_depth=10, n_estimators=100),
    'QDA': QuadraticDiscriminantAnalysis(),
    'Log. Regr.': LogisticRegression(penalty='none', solver='saga', max_iter=10000)
}