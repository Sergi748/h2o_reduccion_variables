# -*- coding: utf-8 -*-
# ! python3 -m pip install -r h2o_requirements.txt
import csv
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
import matplotlib.pyplot as plt
import os
os.system("python3 -m pip install -r h2o_requirements.txt")
import warnings
warnings.filterwarnings('ignore')
from configparser import ConfigParser

import findspark
findspark.init()

# from pysparkling import H2OContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

spark = SparkSession \
    .builder \
    .appName("h2o_reduce_vars") \
    .config("spark.executor.memory", "20g") \
    .config("spark.driver.memory", "10g") \
    .config("spark.executor.cores", "5") \
    .config("spark.executor.instances", "15") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .config("spark.rpc.askTimeout", "1200") \
    .config("spark.scheduler.minRegisteredResourcesRatio", "0.8") \
    .enableHiveSupport() \
    .getOrCreate()

from pysparkling import *
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2OXGBoostEstimator
from tqdm import tqdm

h2oConf = H2OConf()
h2oConf

## Setting H2O Conf for different port
h2oConf.setBasePort(54300)
## Gett H2O Conf Object to see the configuration
h2oConf
## Launching H2O Cluster
hc = H2OContext.getOrCreate(h2oConf)
## Getting H2O Cluster status
h2o.cluster_status()


def get_parameters(pathConfig):
    
    """
    Leemos el fichero de configuracion donde tenemos los distintos parametros para ejecutar la reduccion de variables.
    Parametros:
        pathConfig: Ruta del fichero config. El fichero config debe de estar descargado junto con el fichero 'reduce_vars.py' y el fichero 'run_redvars.py'.
    """
    
    config = ConfigParser()
    config.read(f'{pathConfig}config_reduccion_vars.ini')
    
    project_name =eval(config['reduce_vars']['project_name'])
    pathS3=eval(config['reduce_vars']['pathS3'])
    pathDict=eval(config['reduce_vars']['pathDict'])
    pathFeat=eval(config['reduce_vars']['pathFeat']) 
    fileTrain=eval(config['reduce_vars']['fileTrain'])
    fileTest_sample=eval(config['reduce_vars']['fileTest_sample'])
    model_one_to_one=eval(config['reduce_vars']['model_one_to_one'])
    target=eval(config['reduce_vars']['target'])
    sample=eval(config['reduce_vars']['sample'])
    fractionSample=eval(config['reduce_vars']['fractionSample'])
    auc_limit=eval(config['reduce_vars']['auc_limit'])
    param_h2o=eval(config['reduce_vars']['param_h2o'])

    return project_name, pathS3, pathDict, pathFeat, fileTrain, fileTest_sample, model_one_to_one, target, sample, fractionSample, auc_limit, param_h2o


class read_prepare_files:
        
    def read_dataset(pathS3, fileName, sample=False, fraction=0.1):
        
        def __sample_dataset(sdf, pathS3, nameSaveSample, fraction=0.1):
            print(f'Sampling dataset and save it in path: {pathS3}{nameSaveSample}_h2o')
            sdf_sample = sdf.sample(fraction=fraction, seed=1234)
            sdf_sample.repartition(1).write.mode("overwrite").parquet(f'{pathS3}{nameSaveSample}_h2o')
            return sdf_sample

        if sample == True:
            try:
                sdf_raw = spark.read.parquet(f"{pathS3}{fileName.split('.')[0]}_h2o")
                print(f"Leyendo muestra del fichero {fileName.split('.')[0]}_h2o ya guardado")
                print(f'Numero de registros en la muestra: {sdf_raw.count()}')
            except:
                print(f"Generando muestra del fichero {fileName.split('.')[0]}")
                sdf = spark.read.csv(f'{pathS3}{fileName}', sep='|', inferSchema=True, header=True)
                sdf_raw = __sample_dataset(sdf, pathS3, fileName.split('.')[0], fraction)
                print(f'Numero de registros en la muestra: {sdf_raw.count()}')
        else:
            sdf_raw = spark.read.csv(f'{pathS3}{fileName}', sep='|', inferSchema=True, header=True)
            print(f'Numero de registros en fichero {fileName}: {sdf_raw.count()}')

        return sdf_raw                             

    def read_dict(pathDict):

        scinet_dict = pd.read_csv(f'{pathDict}dictionary.csv', sep="|")
        scinet_dict.columns = scinet_dict.columns.str.lower()
        return scinet_dict

    def prepare_dataset(dataset, scinet_dict):

        scinet_dict['feature'] = scinet_dict['feature'].str.lower()
        categ_vars = scinet_dict[(scinet_dict["used"] == True) & (scinet_dict["type"] == "Categorical")]["feature"].to_list()
        numeric_vars = scinet_dict[(scinet_dict["used"] == True) & (scinet_dict["type"] == "Numerical")]["feature"].to_list()
        # nombres de columnas a minusculas
        dataset = dataset.select([F.col(x).alias(x.lower()) for x in dataset.columns])

        # variables categoricas usadas a strings
        dataset = dataset.select(*(F.col(c).cast("string").alias(c) if (c in categ_vars) else c for c in dataset.columns))
        # variables numericas como double
        dataset = dataset.select(*(F.col(c).cast("double").alias(c) if (c in numeric_vars) else c for c in dataset.columns))
        # varibales a ignorar
        ignore_columns = [col for col in dataset.columns if col not in categ_vars + numeric_vars]

        return dataset, ignore_columns
    
    def datasets_h2o(sdf_train, sdf_test, scinet_dict, target):
        
        sdf_train_prepare, ignore_columns = read_prepare_files.prepare_dataset(sdf_train, scinet_dict)
        sdf_validation_prepare, _ = read_prepare_files.prepare_dataset(sdf_test, scinet_dict)

        h2o_train_sample = hc.asH2OFrame(sdf_train_prepare, "train")
        h2o_validation_sample = hc.asH2OFrame(sdf_validation_prepare, "validation")

        calib_01 = h2o_validation_sample
        h2o_train_sample[target] = h2o_train_sample[target].asfactor()
        
        return h2o_train_sample, h2o_validation_sample, ignore_columns
    
    def get_features(pathFeat, scinet_dict, colsTrain):

        df_feat = pd.read_csv(pathFeat, sep='|')
        df_feat.columns = df_feat.columns.str.lower()
        df_feat.feature = df_feat.feature.str.lower()

        varsCatinFeat = [elem for elem in df_feat.feature.unique().tolist() if elem not in colsTrain]
        varsCatDict = scinet_dict[(scinet_dict.used == True) & (scinet_dict.type == 'Categorical')]['feature'].tolist()

        dictCatinFeat = {}
        for var in varsCatDict:
            for varF in varsCatinFeat:
                if var in varF:
                    dictCatinFeat[varF] = var

        df_feat['feature_final'] = df_feat.feature.map(lambda x: x if x not in varsCatinFeat else dictCatinFeat[x])            
        listFeature = df_feat.feature_final.unique().tolist()

        return listFeature


class train_model:
    
    def train_model_complete(h2o_train_sample, h2o_validation_sample, target, ignore_columns, colsTrain, project_name, param_h2o):
        
        if os.path.exists(project_name) == False:
            os.mkdir(project_name)

        print('##### Ejecutando modelo con todas las variables del diccionario #####')
        ### Importar estimador y Entrenar
        model_h2o_base = H2OXGBoostEstimator(**param_h2o)

        id_train = model_h2o_base.train(
            y=target,
            training_frame=h2o_train_sample,
            validation_frame=h2o_validation_sample,
            ignored_columns=ignore_columns
        )

        res = model_h2o_base.auc(train=True, valid=True, xval=False)
        res["auc_train"] = res.pop("train")
        res["auc_valid"] = res.pop("valid")
        res.update(model_h2o_base.aucpr(train=True, valid=True, xval=False))
        res["prauc_train"] = res.pop("train")
        res["prauc_valid"] = res.pop("valid")
        res.update({"deleted_col": ""})
        res['varimp'] = model_h2o_base.varimp(True).to_dict()

        keys = res.keys()
        with open(f'{project_name}/model_complete.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys, delimiter=";")
            dict_writer.writeheader()

        with open(f'{project_name}/model_complete.csv', 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys, delimiter=";")
            dict_writer.writerow(res)

    def add_rand_var(sdf_train, sdf_test, scinet_dict, target):
    
        sdf_train_new = sdf_train.withColumn("random_variable", F.monotonically_increasing_id())
        sdf_test_new = sdf_test.withColumn("random_variable", F.monotonically_increasing_id())

        scinet_dict_new = scinet_dict.copy()
        scinet_dict_new = scinet_dict_new.append({'feature':'random_variable', 'used':True, 'type':'Numerical', 'missing':-987654321}, ignore_index=True)

        sdf_train_sample_new, ignore_columns_new = read_prepare_files.prepare_dataset(sdf_train_new, scinet_dict_new)
        sdf_validation_sample_new, _ = read_prepare_files.prepare_dataset(sdf_test_new, scinet_dict_new)

        h2o_train_sample_new = hc.asH2OFrame(sdf_train_sample_new, "train")
        h2o_validation_sample_new = hc.asH2OFrame(sdf_validation_sample_new, "validation")

        # convertir en categorica columna de predicciones
        h2o_train_sample_new[target] = h2o_train_sample_new[target].asfactor()

        return h2o_train_sample_new, h2o_validation_sample_new, scinet_dict_new, ignore_columns_new, sdf_train_new.columns

    def __train_h2o(h2o_train_sample_new, h2o_validation_sample_new, ignore_columns_new, varsRemove, i, fileName, target, colsTrain_new, project_name, param_h2o):

        model_h2o_base = H2OXGBoostEstimator(**param_h2o)

        id_train = model_h2o_base.train(
            y=target,
            training_frame=h2o_train_sample_new,
            validation_frame=h2o_validation_sample_new,
            ignored_columns=ignore_columns_new + varsRemove
        )

        res = model_h2o_base.auc(train=True, valid=True, xval=False)
        res["auc_train"] = res.pop("train")
        res["auc_valid"] = res.pop("valid")
        res.update(model_h2o_base.aucpr(train=True, valid=True, xval=False))
        res["prauc_train"] = res.pop("train")
        res["prauc_valid"] = res.pop("valid")
        res.update({"deleted_col": ""})
        res['varimp'] = model_h2o_base.varimp(True).to_dict()
        res['number_vars_used'] = len(h2o_train_sample_new.columns) - len(ignore_columns_new + varsRemove) 
        res['vars_removed'] = ignore_columns_new + varsRemove
        res['number_vars_removed'] = len(ignore_columns_new + varsRemove) 

        keys = res.keys()
        if i == 0:
            with open(f'{project_name}/model_variable_reduction_random_variable{fileName}.csv', 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys, delimiter=";")
                dict_writer.writeheader()

        with open(f'{project_name}/model_variable_reduction_random_variable{fileName}.csv', 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys, delimiter=";")
            dict_writer.writerow(res)
            
    def __extract_info(fileName, removeCharacter, project_name):

        df_red_rand_var = pd.read_csv(f'{project_name}/model_variable_reduction_random_variable{fileName}.csv', sep=';')
        auc_train = df_red_rand_var.loc[len(df_red_rand_var)-1 , 'auc_train']
        vars_used = df_red_rand_var.loc[len(df_red_rand_var)-1 , 'number_vars_used']
        print(f'Number of variables used: {vars_used}')
        dfFeats = pd.DataFrame.from_dict(eval(df_red_rand_var.loc[len(df_red_rand_var)-1, 'varimp']))
        indexRandVar = dfFeats[dfFeats.variable == 'random_variable'].index[0]
        varsRemove_randVar = dfFeats[dfFeats.index > indexRandVar]['variable'].tolist()
        if removeCharacter == True:
            varsRemove_randVar_final = list(set([elem.split('.')[0] if elem.__contains__('.') else elem for elem in varsRemove_randVar]))
        else:
            varsRemove_randVar_final = list(set([elem for elem in varsRemove_randVar if elem.__contains__('.') != True]))

        return auc_train, varsRemove_randVar_final    
    
    def run_reduccion_variables(h2o_train, h2o_validation, ignore_columns, target, colsTrain_new, project_name, auc_limit, param_h2o):

        train_model.__train_h2o(h2o_train, h2o_validation, ignore_columns, [], 0, '_with_categorical', target, colsTrain_new, project_name, param_h2o)
        auc_train, varsRemove_randVar_final = train_model.__extract_info('_with_categorical', True, project_name)

        varsRemove = varsRemove_randVar_final
        i = 1

        while (auc_train >= auc_limit) and (len(varsRemove_randVar_final) >= 1):
            train_model.__train_h2o(h2o_train, h2o_validation, ignore_columns, varsRemove, i, '_with_categorical', target, colsTrain_new, project_name, param_h2o)
            i = i+1
            auc_train, varsRemove_randVar_final = train_model.__extract_info('_with_categorical', True, project_name)
            varsRemove = varsRemove + varsRemove_randVar_final
            print(f'auc_train: {auc_train} \nNumber of variables removed: {len(varsRemove_randVar_final)} \nTotal number of variables removed: {len(varsRemove)}')

    def train_model_one_to_one(h2o_train_sample, h2o_validation_sample, target, listFeature, colsTrain, project_name, param_h2o):
    
        for i, var in enumerate(listFeature, start=1):

            if (i % 10 == 0) or (i == 1):
                print(f'Modelo {i} de {len(listFeature)}')
            varsUsed = listFeature[:i]
            ignore_columns = [elem for elem in colsTrain if elem not in varsUsed]

            model_h2o_base = H2OXGBoostEstimator(**param_h2o)
            id_train = model_h2o_base.train(
                y=target,
                training_frame=h2o_train_sample,
                validation_frame=h2o_validation_sample,
                ignored_columns=ignore_columns
            )

            res = model_h2o_base.auc(train=True, valid=True, xval=False)
            res["auc_train"] = res.pop("train")
            res["auc_valid"] = res.pop("valid")
            res.update(model_h2o_base.aucpr(train=True, valid=True, xval=False))
            res["prauc_train"] = res.pop("train")
            res["prauc_valid"] = res.pop("valid")
            res.update({"deleted_col": ""})
            res['varimp'] = {} if i == 1 else model_h2o_base.varimp(True).to_dict()
            res['newVar_used'] = var
            res['number_vars_used'] = len(varsUsed)
            res['vars_used'] = varsUsed
            res['number_vars_removed'] = len(ignore_columns)

            keys = res.keys()
            if i == 1:
                with open(f'{project_name}/model_one_to_one.csv', 'w', newline='') as output_file:
                    dict_writer = csv.DictWriter(output_file, keys, delimiter=";")
                    dict_writer.writeheader()

            with open(f'{project_name}/model_one_to_one.csv', 'a', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys, delimiter=";")
                dict_writer.writerow(res)
