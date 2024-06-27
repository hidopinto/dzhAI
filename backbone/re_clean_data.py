import pandas as pd
import numpy as np

from backbone.predict_mortality import load_data


COLS_TO_NUMERIC = ['HB', 'TLC', 'PLATELETS', 'GLUCOSE', 'UREA', 'CREATININE', 'BNP', 'EF', 'CHEST INFECTION']
CAT_COLS = ['GENDER', 'RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD']
BAD_COLS = ['month year']


def main():
    data = pd.read_csv('../Data/cleaned_admission_data.csv', index_col=0)

    # removes bad cols
    data = data.drop(columns=BAD_COLS)

    # fix bad values at columns
    data = data.replace('EMPTY', np.nan)

    # remove a raw of bad maintenance
    data = data[data['CHEST INFECTION'] != '\\']

    for col in COLS_TO_NUMERIC:
        data[col] = pd.to_numeric(data[col])

    # handle cat cols
    cat_data = pd.get_dummies(data[CAT_COLS])
    data = pd.concat([data.drop(columns=CAT_COLS), cat_data])

    # final outcome col
    outcome = data['OUTCOME']
    data['OUTCOME'].isin(['DISCHARGE', 'EXPIRY'])
    data['OUTCOME'] = (outcome == 'EXPIRY').astype('int')

    print('saving re-cleaned data with cat cols')
    data.to_csv('~../Data/recleaned_admission_data.csv')


if __name__ == '__main__':
    main()
