from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


classifier_dict = {
    'Log. Regr.': LogisticRegression(penalty='none', solver='saga', max_iter=10000),
    'QDA': QuadraticDiscriminantAnalysis(),
    'NN': KNeighborsClassifier(),
    'XGBoost \n (d3, n1000)': XGBClassifier(n_estimators=1000),
    'XGBoost \n (d5, n1000)': XGBClassifier(max_depth=5, n_estimators=1000),
    'XGBoost \n (d10, n100)': XGBClassifier(max_depth=10),
    'MLP1': MLPClassifier((64, 32, 32), activation='tanh', alpha=0, max_iter=25000)
}
