# +
# Manera de ejecutarse:
# nohup python3 main.py &
# -

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from h2o_reduccion_variables import *

project_name, pathS3, pathDict, pathFeat, fileTrain, fileTest_sample, model_one_to_one, target, sample, fractionSample, auc_limit, param_h2o = get_parameters('./')
project_name, pathS3, pathDict, pathFeat, fileTrain, fileTest_sample, model_one_to_one, target, sample, fractionSample, auc_limit, param_h2o

sdf_train = read_prepare_files.read_dataset(pathS3, fileTrain, sample, fractionSample)
sdf_test = read_prepare_files.read_dataset(pathS3, fileTest_sample, sample, fractionSample)
scinet_dict = read_prepare_files.read_dict(pathDict)

h2o_train_sample, h2o_validation_sample, ignore_columns = read_prepare_files.datasets_h2o(sdf_train, sdf_test, scinet_dict, target)

train_model.train_model_complete(h2o_train_sample, h2o_validation_sample, target, ignore_columns, sdf_train.columns, project_name, param_h2o)

if model_one_to_one == True:
    listFeature = read_prepare_files.get_features(pathFeat, scinet_dict, sdf_train.columns)
    train_model.train_model_one_to_one(h2o_train_sample, h2o_validation_sample, target, listFeature, sdf_train.columns, project_name, param_h2o)

h2o_train_sample_new, h2o_validation_sample_new, scinet_dict_new, ignore_columns_new, colsTrain_new = train_model.add_rand_var(sdf_train, sdf_test, scinet_dict, target)

train_model.run_reduccion_variables(h2o_train_sample_new, h2o_validation_sample_new, ignore_columns_new, target, colsTrain_new, project_name, auc_limit, param_h2o)

spark.stop()


