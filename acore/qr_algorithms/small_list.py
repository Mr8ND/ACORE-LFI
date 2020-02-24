classifier_cde_dict = {
    'xgb_d3_n100': ('xgb', {'max_depth': 3, 'n_estimators': 100}),
    'xgb_d5_n250': ('xgb', {'max_depth': 5, 'n_estimators': 250}),
    'RF100': ('rf', {'n_estimators': 100}),
    'pytorch': ('pytorch', {'epochs': 10, 'batch_size': 50}, {'neur_shapes': (64, 64)})
}
