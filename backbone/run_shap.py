import joblib
import shap

from backbone.predict_mortality import load_data, get_train_data, OUTCOME_COL, BLOOD_TEST_COLUMNS


def calc_shap(model, data):
    x, y = get_train_data(data, [OUTCOME_COL])

    shap_values = shap.TreeExplainer(model).shap_values(x)

    return shap_values


def main():
    print('loading data')
    data = load_data()
    data_without_blood_tests = data.drop(columns=BLOOD_TEST_COLUMNS)

    print('loading pred model wo blood tests')
    pred_wo_bt_model = joblib.load('/home/yehduap/Desktop/SegalLab/dzhAI/output/models/pred_wo_bt_model.joblib')

    print('loading pred model with blood tests')
    pred_bt_model = joblib.load('/home/yehduap/Desktop/SegalLab/dzhAI/output/models/pred_bt_model.joblib')

    print('calculating shap values wo blood test')
    shap_values_wo_bt = calc_shap(pred_wo_bt_model, data_without_blood_tests)
    print('saving shap values wo blood test')
    joblib.dump(shap_values_wo_bt, '/home/yehduap/Desktop/SegalLab/dzhAI/output/models/shap_values_wo_bt.joblib')

    print('calculating shap values with blood test')
    shap_values_bt = calc_shap(pred_bt_model, data)
    print('saving shap values with blood test')
    joblib.dump(shap_values_bt, '/home/yehduap/Desktop/SegalLab/dzhAI/output/models/shap_values_bt.joblib')


if __name__ == '__main__':
    main()
