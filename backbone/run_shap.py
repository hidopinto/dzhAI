import joblib
import matplotlib.pyplot as plt
import shap

from backbone.predict_mortality import load_data, OUTCOME_COL, BLOOD_TEST_COLUMNS


def calc_shap(model, data, suffix):
    x = data.drop(columns=[OUTCOME_COL])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    shap.summary_plot(shap_values, x, plot_type='bar', show=False)
    plt.savefig(f'../output/figures/shap_summary_plot_{suffix}.png', dpi=300, bbox_inches='tight')
    plt.clf()
    shap_explainer_obj = explainer(x)
    shap.plots.beeswarm(shap_explainer_obj, show=False, plot_size=(10, 10))

    # plt.savefig(f'../output/figures/shap_beeswarm_{suffix}.png', dpi=300, bbox_inches='tight')
    # plt.clf()
    #
    # shap.plots.scatter(shap_explainer_obj[:, "daily_order_of_arrival"], show=False)
    #
    # plt.savefig(f'../output/figures/shap_scatter_{suffix}.png', dpi=300, bbox_inches='tight')
    # plt.clf()


    shap.plots.waterfall(shap_explainer_obj[10], show=False)
    plt.savefig(f'../output/figures/shap_waterfall_{suffix}.png', dpi=300, bbox_inches='tight')
    plt.clf()

    return explainer, shap_values


def main():
    print('loading data')
    data = load_data()
    data_without_blood_tests = data.drop(columns=BLOOD_TEST_COLUMNS)

    print('loading pred model wo blood tests')
    pred_wo_bt_model = joblib.load('../output/models/pred_wo_bt_model.joblib')

    print('loading pred model with blood tests')
    pred_bt_model = joblib.load('../output/models/pred_bt_model.joblib')

    print('calculating shap values wo blood test')
    explainer_wo_bt, shap_values_wo_bt = calc_shap(pred_wo_bt_model, data_without_blood_tests, 'wo_bt')
    print('saving shap values wo blood test')
    joblib.dump(shap_values_wo_bt, '../output/models/shap_values_wo_bt.joblib')

    print('calculating shap values with blood test')
    explainer_bt, shap_values_bt = calc_shap(pred_bt_model, data, 'bt')
    print('saving shap values with blood test')
    joblib.dump(shap_values_bt, '../output/models/shap_values_bt.joblib')


if __name__ == '__main__':
    main()
