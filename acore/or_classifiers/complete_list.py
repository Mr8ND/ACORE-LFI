from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier


classifier_dict = {
    'lg': LogisticRegression(penalty='none', solver='saga', max_iter=10000),
    'qda': QuadraticDiscriminantAnalysis(),
    'knn': KNeighborsClassifier(),
    'rf_n200': RandomForestClassifier(n_estimators=200),
    'xgb_d3_n1000': XGBClassifier(n_estimators=1000),
    'xgb_d3_n100': XGBClassifier(n_estimators=100),
    'xgb_d3_n500': XGBClassifier(n_estimators=500),
    'xgb_d5_n1000': XGBClassifier(max_depth=5, n_estimators=1000),
    'xgb_d5_n100': XGBClassifier(max_depth=5, n_estimators=100),
    'xgb_d5_n500': XGBClassifier(max_depth=5, n_estimators=500),
    'xgb_d10_n100': XGBClassifier(max_depth=10),
    'xgb_d10_n500': XGBClassifier(max_depth=10, n_estimators=500),
    'mlp': MLPClassifier(alpha=0, max_iter=1000),
    'gauss_proc_rbf1': GaussianProcessClassifier(RBF(1.0)),
    'gauss_proc_rbf.1': GaussianProcessClassifier(RBF(.1)),
    'gauss_proc_rbf.5': GaussianProcessClassifier(RBF(.5)),
    'gauss_proc_.5rbf.1': GaussianProcessClassifier(0.5 * RBF(.1)),
    'mlp1t': MLPClassifier((32, 16), activation='tanh', alpha=0, max_iter=25000),
    'mlp1r': MLPClassifier((32, 16), activation='relu', alpha=0, max_iter=25000),
    'mlp2t': MLPClassifier((64, 32, 32), activation='tanh', alpha=0, max_iter=25000),
    'mlp2r': MLPClassifier((64, 32, 32), activation='relu', alpha=0, max_iter=25000),
    'mlp3t': MLPClassifier((128, 64, 32), activation='tanh', alpha=0, max_iter=25000),
    'mlp2r_a': MLPClassifier((128, 64, 32), activation='relu', alpha=0, max_iter=25000),
    'mlp3t_a': MLPClassifier((128, 32, 32), activation='tanh', alpha=0, max_iter=25000),
    'mlp3r': MLPClassifier((128, 32, 32), activation='relu', alpha=0, max_iter=25000),
    'mlp4t': MLPClassifier((128, 64, 32,  32), activation='tanh', alpha=0, max_iter=25000),
    'mlp4r': MLPClassifier((128, 64, 32,  32), activation='relu', alpha=0, max_iter=25000),
    'mlp5t': MLPClassifier((128, 64, 64, 32), activation='tanh', alpha=0, max_iter=25000),
    'mlp5r': MLPClassifier((128, 64, 64, 32), activation='relu', alpha=0, max_iter=25000),
    'mlp6t': MLPClassifier((256, 128, 64, 32), activation='tanh', alpha=0, max_iter=25000),
    'mlp6r': MLPClassifier((256, 128, 64, 32), activation='relu', alpha=0, max_iter=25000),
    'mlp7r': MLPClassifier((512, 256, 64, 32, 32), activation='relu', alpha=0, max_iter=25000),
    'mlp7t': MLPClassifier((512, 256, 64, 32, 32), activation='tanh', alpha=0, max_iter=25000),
    'mlp8r': MLPClassifier((1024, 512, 256, 64, 32, 32), activation='relu', alpha=0, max_iter=25000),
    'mlp8t': MLPClassifier((1024, 512, 256, 64, 32, 32), activation='tanh', alpha=0, max_iter=25000)
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
