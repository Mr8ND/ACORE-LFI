from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier


classifier_dict = {
    'Log. Regr.': LogisticRegression(penalty='none', solver='saga', max_iter=10000),
    'QDA': QuadraticDiscriminantAnalysis(),
    'NN': KNeighborsClassifier(),
    'RF_n200': RandomForestClassifier(n_estimators=200),
    'XGBoost \n (d3, n1000)': XGBClassifier(n_estimators=1000),
    'XGBoost \n (d3, n100)': XGBClassifier(n_estimators=100),
    'XGBoost \n (d3, n500)': XGBClassifier(n_estimators=500),
    'XGBoost \n (d5, n1000)': XGBClassifier(max_depth=5, n_estimators=1000),
    'XGBoost \n (d5, n100)': XGBClassifier(max_depth=5, n_estimators=100),
    'XGBoost \n (d5, n500)': XGBClassifier(max_depth=5, n_estimators=500),
    'XGBoost \n (d10, n100)': XGBClassifier(max_depth=10),
    'XGBoost \n (d10, n500)': XGBClassifier(max_depth=10, n_estimators=500),
    'MLP': MLPClassifier(alpha=0, max_iter=1000),
    'Gauss_Proc1': GaussianProcessClassifier(RBF(1.0)),
    'Gauss_Proc2': GaussianProcessClassifier(RBF(.1)),
    'Gauss_Proc3': GaussianProcessClassifier(RBF(.5)),
    'Gauss_Proc4': GaussianProcessClassifier(0.5 * RBF(.1)),
    'MLP1t': MLPClassifier((32, 16), activation='tanh', alpha=0, max_iter=25000),
    'MLP1': MLPClassifier((32, 16), activation='relu', alpha=0, max_iter=25000),
    'MLP2t': MLPClassifier((64, 32, 32), activation='tanh', alpha=0, max_iter=25000),
    'MLP2': MLPClassifier((64, 32, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP3t': MLPClassifier((128, 64, 32), activation='tanh', alpha=0, max_iter=25000),
    'MLP3': MLPClassifier((128, 64, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP3t_a': MLPClassifier((128, 32, 32), activation='tanh', alpha=0, max_iter=25000),
    'MLP3_a': MLPClassifier((128, 32, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP4t': MLPClassifier((128, 64, 32,  32), activation='tanh', alpha=0, max_iter=25000),
    'MLP4': MLPClassifier((128, 64, 32,  32), activation='relu', alpha=0, max_iter=25000),
    'MLP5t': MLPClassifier((128, 64, 64, 32), activation='tanh', alpha=0, max_iter=25000),
    'MLP5': MLPClassifier((128, 64, 64, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP6t': MLPClassifier((256, 128, 64, 32), activation='tanh', alpha=0, max_iter=25000),
    'MLP6': MLPClassifier((256, 128, 64, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP7': MLPClassifier((512, 256, 64, 32, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP7t': MLPClassifier((512, 256, 64, 32, 32), activation='tanh', alpha=0, max_iter=25000),
    'MLP8': MLPClassifier((1024, 512, 256, 64, 32, 32), activation='relu', alpha=0, max_iter=25000),
    'MLP8t': MLPClassifier((1024, 512, 256, 64, 32, 32), activation='tanh', alpha=0, max_iter=25000)
}

classifier_conv_dict = {
    'nn': 'NN',
    'qda': 'QDA',
    'lg': 'Log. Regr.',
    'rf_n200': 'RF_n200',
    'xgb_d3_n1000': 'XGBoost \n (d3, n1000)',
    'xgb_d3_n500': 'XGBoost \n (d3, n500)',
    'xgb_d3_n100': 'XGBoost \n (d3, n100)',
    'xgb_d5_n1000': 'XGBoost \n (d5, n1000)',
    'xgb_d5_n100': 'XGBoost \n (d5, n100)',
    'xgb_d5_n500': 'XGBoost \n (d5, n500)',
    'xgb_d10_n100': 'XGBoost \n (d10, n100)',
    'xgb_d10_n500': 'XGBoost \n (d10, n500)',
    'gp1': 'Gauss_Proc1',
    'gp2': 'Gauss_Proc2',
    'gp3': 'Gauss_Proc3',
    'gp4': 'Gauss_Proc4',
    'mlp1': 'MLP1',
    'mlp2': 'MLP2',
    'mlp3': 'MLP3',
    'mlp4': 'MLP4',
    'mlp5': 'MLP5',
    'mlp6': 'MLP6',
    'mlp7': 'MLP7',
    'mlp8': 'MLP8',
    'mlp1t': 'MLP1t',
    'mlp2t': 'MLP2t',
    'mlp3t': 'MLP3t',
    'mlp4t': 'MLP4t',
    'mlp5t': 'MLP5t',
    'mlp6t': 'MLP6t',
    'mlp7t': 'MLP7t',
    'mlp8t': 'MLP8t',
    'mlp3_a': 'MLP3_a',
    'mlp3t_a': 'MLP3t_a',
}
