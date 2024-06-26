import pandas as pd
import lightgbm as lgb
import joblib

from backbone.lgbm.train import r2_score_wo_nans
from backbone.train_helper import train_multi_model
from backbone.train_utils.trainable_factory import OptunaLGBMFactory
from backbone.conf import conf as default_conf


OUTCOME_COL = 'OUTCOME'
BLOOD_TEST_COLUMNS = []


### load data
def load_data() -> pd.DataFrame:
    return pd.read_csv('~/Desktop/SegalLab/dzhAI/Data/cleaned_admission_data.csv')


### conf
def create_conf(default_mortality_conf=default_conf):
    conf_copy = default_mortality_conf.copy()
    # TODO: create conf, set run_cross_validation to False

    conf_copy['train']['trainable']['args']['param_grid'] = {}
    conf_copy['train']['trainable']['args']['scoring'] = r2_score_wo_nans

    return conf_copy


### train funcs
def train_with_blood_tests(data: pd.DataFrame, conf):
    trainable_factory = OptunaLGBMFactory(conf, model_class=lgb.LGBMRegressor)
    trainable_model = trainable_factory.create()

    x = data.drop(columns=[OUTCOME_COL])
    y = data[[OUTCOME_COL]]
    models, _ = train_multi_model(x, y, trainable_model, conf)

    return models


def train_without_blood_tests(data: pd.DataFrame, conf):
    data_without_blood_tests = data.drop(columns=BLOOD_TEST_COLUMNS)
    trainable_factory = OptunaLGBMFactory(conf, model_class=lgb.LGBMRegressor)
    trainable_model = trainable_factory.create()

    x = data_without_blood_tests.drop(columns=[OUTCOME_COL])
    y = data_without_blood_tests[[OUTCOME_COL]]
    models, _ = train_multi_model(x, y, trainable_model, conf)

    return models

# TODO: add SHAP


def main():
    conf = create_conf()

    print('loading data')
    data = load_data()

    print('training pred model wo blood tests')
    pred_wo_bt_model = train_without_blood_tests(data, conf)[OUTCOME_COL]
    print('saving pred model wo blood tests')
    joblib.dump(pred_wo_bt_model, '~/Desktop/SegalLab/dzhAI/output/models/pred_wo_bt_model.joblib')

    print('training pred model with blood tests')
    pred_bt_model = train_with_blood_tests(data, conf)[OUTCOME_COL]
    print('saving pred model with blood tests')
    joblib.dump(pred_bt_model, '~/Desktop/SegalLab/dzhAI/output/models/pred_bt_model.joblib')


if __name__ == '__main__':
    main()
