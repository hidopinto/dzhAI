from backbone.train_utils.trainable_factory import GridSearchTrainableFactory


conf = {
    'lgbm': {
        'train': {
            'direction': 'minimize',
            'n_trials': 50,
            'early_stopping': {
                'stopping_rounds': 100
            },
            'num_cores': 8
        }
    },
    'train': {
        'cv_params': {
            'cv': 5,
            'n_jobs': -1,
            'scoring': ['r2', 'neg_mean_squared_error'],
            'verbose': 0,
            'return_train_score': False
        },
        'run_cross_validation': False,
        'trainable': {
            'factory_class': GridSearchTrainableFactory,
            'args': {
                'param_grid': {},
                'cv': 3,
                'n_jobs': 3,
                'verbose': 1,
                'scoring': None,
                'refit': True
            }
        }
    },
    'jobs_path': '/net/mraid20/export/jasmine/yehudap/'
}
