from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

regressor_dict = {
    "linear_reg": LinearRegression(fit_intercept=True),

    'xgb_d3_n100': XGBRegressor(n_estimators=100, max_depth=3),
    'xgb_d3_n500': XGBRegressor(n_estimators=500, max_depth=3),
    'xgb_d3_n1000': XGBRegressor(n_estimators=1000, max_depth=3),

    'xgb_d5_n100': XGBRegressor(n_estimators=100, max_depth=5),
    'xgb_d5_n500': XGBRegressor(n_estimators=500, max_depth=5),
    'xgb_d5_n1000': XGBRegressor(n_estimators=1000, max_depth=5),

    'xgb_d10_n100': XGBRegressor(n_estimators=100, max_depth=10),
    'xgb_d10_n500': XGBRegressor(n_estimators=500, max_depth=10),
    'xgb_d10_n1000': XGBRegressor(n_estimators=1000, max_depth=10),

    'mlp1i': MLPRegressor(hidden_layer_sizes=(32, 16),
                          activation='identity', alpha=0, max_iter=25000, solver="adam")
}
