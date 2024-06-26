##########################
# code from the notebook #
##########################
import re


def mse_wo_nans(y_true, y_pred):
    from numpy import isnan
    from sklearn.metrics import mean_squared_error

    higher_is_better = False

    indices = ~isnan(y_true)
    loss = mean_squared_error(y_true[indices], y_pred[indices])

    return 'mse_wo_nans', loss, higher_is_better


def r2_score_wo_nans(y_true, y_pred):
    from numpy import isnan
    from sklearn.metrics import r2_score

    higher_is_better = True

    indices = ~isnan(y_true)
    loss = r2_score(y_true[indices], y_pred[indices])

    return 'r2_score_wo_nans', loss, higher_is_better


def rename_col_names_for_lgbm(df):
    return df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_(). -]+', '', x))


"""
Code snippet examples:

Create trainable snippet:
        trainable_factory = OptunaLGBMFactory(conf, lgb_val)
        trainable_model = trainable_factory.create()

Train trainable snippet:
        train_multi_model(train_x, train_y, trainable_model, train_conf, queue)

Train trainable with queue snippet:
        trained_model_ticket = q.method(train_multi_model, (train_x, train_y, trainable_model, train_conf, queue))
        trained_model = q.waitforresult(trained_model_ticket)
"""
