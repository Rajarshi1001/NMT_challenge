import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

TRAIN_FILE = "train_data2.json"
VALIDATION_FILE = "test_data2_final.json"
TEST_FILE = "test_data1_final.json"

# Defining lists for each language pair
train_bng_ids, train_eb, train_bng = [],[],[]
train_guj_ids, train_eg, train_guj = [],[],[]
train_hin_ids, train_eh, train_hin = [],[],[]
train_kn_ids, train_ek, train_kn = [],[],[]
train_ml_ids, train_em, train_ml = [],[],[]
train_tm_ids, train_et, train_tm = [],[],[]
train_tl_ids, train_etl, train_tl = [],[],[]
train_ml_ids, train_eml, train_ml = [],[],[]
val_ids = []

def collectData():

    with open(TRAIN_FILE, "r") as file:
        data = json.load(file)
        for language_type, lang_data in data.items():
            # print(lang_data)
            for data_type, data_entries in lang_data.items():
                print(data_type)

                for ent_id, ent_data in data_entries.items():
                    source_sen = ent_data["source"]
                    target_sen = ent_data["target"]
                    if language_type == "English-Bengali":  
                        train_eb.append(source_sen)
                        train_bng.append(target_sen)
                        train_bng_ids.append(ent_id)
                    if language_type == "English-Gujarati":  
                        train_eg.append(source_sen)
                        train_guj.append(target_sen)
                        train_guj_ids.append(ent_id)
                    if language_type == "English-Hindi":  
                        train_eh.append(source_sen)
                        train_hin.append(target_sen)
                        train_hin_ids.append(ent_id)
                    if language_type == "English-Kannada":  
                        train_ek.append(source_sen)
                        train_kn.append(target_sen)
                        train_kn_ids.append(ent_id)
                    if language_type == "English-Tamil":  
                        train_et.append(source_sen)
                        train_tm.append(target_sen)
                        train_tm_ids.append(ent_id)
                    if language_type == "English-Telgu":  
                        train_etl.append(source_sen)
                        train_tl.append(target_sen)
                        train_tl_ids.append(ent_id)
                    if language_type == "English-Malayalam":  
                        train_eml.append(source_sen)
                        train_ml.append(target_sen)
                        train_ml_ids.append(ent_id)

    DATA_DIR = "phase2_data"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    eb_df = pd.DataFrame({"entry_id" : train_bng_ids, "english" : train_eb, "bengali": train_bng})
    eg_df = pd.DataFrame({"entry_id" : train_guj_ids, "english" : train_eg, "gujarati": train_guj})
    eh_df = pd.DataFrame({"entry_id" : train_hin_ids, "english" : train_eh, "hindi": train_hin})
    ek_df = pd.DataFrame({"entry_id" : train_kn_ids, "english" : train_ek, "kannada": train_kn})
    etm_df = pd.DataFrame({"entry_id" : train_tm_ids, "english" : train_et, "tamil": train_tm})
    etl_df = pd.DataFrame({"entry_id" : train_tl_ids, "english" : train_etl, "telegu": train_tl})
    eml_df = pd.DataFrame({"entry_id" : train_ml_ids, "english" : train_eml, "malayalam" : train_ml})

    eb_df.to_csv(os.path.join(DATA_DIR,"bengali.csv"))
    eg_df.to_csv(os.path.join(DATA_DIR,"gujarati.csv"))
    eh_df.to_csv(os.path.join(DATA_DIR,"hindi.csv"))
    ek_df.to_csv(os.path.join(DATA_DIR,"kannada.csv"))
    etl_df.to_csv(os.path.join(DATA_DIR,"telugu.csv"))
    etm_df.to_csv(os.path.join(DATA_DIR,"tamil.csv"))
    eml_df.to_csv(os.path.join(DATA_DIR,"malayalam.csv"))

def fetchtestData():
    

    DATA_DIR = "phase2_testdata"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    with open(VALIDATION_FILE, "r") as file:
        val_data = json.load(file)
        for lang_type, lang_data in val_data.items():
            print(lang_type)
            for data_type, data_entries in lang_data.items():
                ids, en_sns = [], []
                for ent_id, ent_data in data_entries.items():
                    val_ids.append(ent_id)
                    ids.append(ent_id)
                    en_sns.append(ent_data["source"])
    
                valdf = pd.DataFrame({"id" : ids, "english" : en_sns})
                valdf.to_csv(os.path.join(DATA_DIR,"test{}.csv".format(lang_type)))
    print(len(val_ids))

def merge():
    DATA_DIR = "models/transformer_translations"
    RESULTS = "answer.csv"

    if os.path.exists(os.path.join(DATA_DIR, RESULTS)):
        os.remove(os.path.join(DATA_DIR,RESULTS))
    val_ids = []

    with open(VALIDATION_FILE, "r") as file:
        val_data = json.load(file)
        for lang_type, lang_data in val_data.items():
            print(lang_type)
            for data_type, data_entries in lang_data.items():
                ids, en_sns = [], []
                for ent_id, ent_data in data_entries.items():
                    val_ids.append(int(ent_id))

    print(len(val_ids))
    # print(val_ids)

    # Reading all the dataframes
    files = os.listdir(DATA_DIR)
    files = [file for file in files if file.endswith(".csv")]
    print(files)
    data = [pd.read_csv(os.path.join(DATA_DIR, file)) for file in files]
    res = pd.concat(data, axis=0, ignore_index=False)
    res.drop(["Unnamed: 0", "english"], axis=1, inplace=True)
    res = res.rename(columns={"id": "ID", "translated": "Translated"})
    res_sorted = res[res["ID"].isin(val_ids)].sort_values(by="ID")
    res_sorted.to_csv(os.path.join(DATA_DIR, RESULTS), sep="\t", index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar='"', escapechar='\\')


if __name__ == "__main__":
    # collectData()
    # fetchtestData()
    merge()
