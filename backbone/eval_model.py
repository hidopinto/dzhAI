import joblib
import shap
from sklearn.metrics import r2_score, auc, roc_auc_score, mean_squared_error

from backbone.predict_mortality import load_data, get_train_data, OUTCOME_COL, BLOOD_TEST_COLUMNS


def main():
    print('loading data')
    data = load_data()
    data_without_blood_tests = data.drop(columns=BLOOD_TEST_COLUMNS)

    x_train_bt, x_test_bt, y_train_bt, y_test_bt = get_train_data(data, [OUTCOME_COL])
    x_train_wo_bt, x_test_wo_bt, y_train_wo_bt, y_test_wo_bt = get_train_data(data_without_blood_tests, [OUTCOME_COL])

    print('loading pred model wo blood tests')
    pred_wo_bt_model = joblib.load('../output/models/pred_wo_bt_model.joblib')

    print('loading pred model with blood tests')
    pred_bt_model = joblib.load('../output/models/pred_bt_model.joblib')

    preds_wo_bt = pred_wo_bt_model.predict(x_test_wo_bt)
    preds_bt = pred_bt_model.predict(x_test_bt)

    r2_wo_bt = r2_score(y_test_wo_bt, preds_wo_bt)
    auc_wo_bt = roc_auc_score(y_test_wo_bt, preds_wo_bt)
    mse_wo_bt = mean_squared_error(y_test_wo_bt, preds_wo_bt)
    print(f'wo blood test | test | r2: {r2_wo_bt} auc: {auc_wo_bt} mse: {mse_wo_bt}')

    r2_bt = r2_score(y_test_bt, preds_bt)
    auc_bt = roc_auc_score(y_test_bt, preds_bt)
    mse_bt = mean_squared_error(y_test_bt, preds_bt)
    print(f'blood test | test | r2: {r2_bt} auc: {auc_bt} mse: {mse_bt}')


if __name__ == '__main__':
    main()
