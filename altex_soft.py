import pandas as pd
from sys import argv
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def define_col_types(df):

    for letter in ['u', 'g', 'i', 'r', 'z']:
        for i in range(6):
            column_name = '{}_{}'.format(letter, i)
            df[column_name] = df[column_name].apply(pd.to_numeric, downcast='float', errors='coerce')

        column_name = '{}_6'.format(letter)
        df[column_name] = df[column_name].apply(pd.to_numeric, downcast='signed', errors='coerce')
    for letter in ['rowv', 'colv']:
        df[letter] = df[column_name].apply(pd.to_numeric, downcast='float', errors='coerce')
        df['clean'] = df['clean'].astype('bool')
    return df


if __name__ == '__main__':
    train_data_path = argv[1]
    unlabeled_data_path = argv[2]
    test_data_path = argv[3]
    predictions_data_path = argv[4]

    train_data=pd.read_csv(train_data_path, index_col=0)
    unlabeled_data = pd.read_csv(unlabeled_data_path, index_col=0)
    test_data=pd.read_csv(test_data_path, index_col=0)

    train_data=define_col_types(train_data)
    test_data = define_col_types(test_data)

    columns_to_drop = ['u_1', 'g_1', 'r_1', 'i_1', 'z_1', 'u_2', 'g_2', 'r_2', 'i_2', 'z_2', 'r_0', 'i_0', 'z_0', 'g_0',
                       'colv', 'z_6']

    train_data.drop(columns=columns_to_drop, inplace=True)
    train_data=train_data.replace('na',np.nan)
    train_data = train_data.fillna(train_data.mean())

    test_data.drop(columns=columns_to_drop, inplace=True)
    test_data=test_data.replace('na',np.nan)
    test_data = test_data.fillna(train_data.mean())


    X_train=train_data[['ra', 'dec', 'u_0', 'clean', 'rowc', 'colc', 'rowv', 'u_3', 'g_3', 'r_3', 'i_3', 'z_3', 'u_4', 'g_4',
                  'r_4', 'i_4', 'z_4', 'u_5', 'g_5', 'r_5', 'i_5', 'z_5', 'u_6', 'g_6', 'r_6', 'i_6',]]


    y_train=train_data['class']

    X_test = test_data[
        ['ra', 'dec', 'u_0', 'clean', 'rowc', 'colc', 'rowv', 'u_3', 'g_3', 'r_3', 'i_3', 'z_3', 'u_4', 'g_4',
         'r_4', 'i_4', 'z_4', 'u_5', 'g_5', 'r_5', 'i_5', 'z_5', 'u_6', 'g_6', 'r_6', 'i_6', ]]


    rfc = RandomForestClassifier(n_estimators=100,random_state=0,max_depth=3).fit(X_train,y_train)

    y_pred=rfc.predict(X_test)
    X_test['prediction'] = y_pred
    X_test_result= X_test[['prediction']]
    X_test_result.to_csv(predictions_data_path )
