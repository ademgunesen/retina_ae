import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import math
import json
import pickle
import datetime
import yagmail
import tensorflow as tf
import csv

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import load_img

def save_var(var, file_name):
    '''
    Saves any type of variable with the given filename(can be a path)
    '''
    out_file = open(file_name,'wb')
    pickle.dump(var,out_file)
    out_file.close()

def read_var(file_name):
    infile = open(file_name,'rb')
    var = pickle.load(infile)
    infile.close()
    return var

def save_model(model, name):
    model_json = model.to_json()
    with open(name+'.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(name+'.h5')
    print("Saved model to disk")

def load_model(name):
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name+'.h5')
    print("Loaded model from disk")
    return loaded_model

def make_subfolder(dirname,parent_path):
    path = os.path.join(parent_path, dirname)
    os.mkdir(path)
    print("Directory '%s' created" %dirname)
    return path + '/'

def make_log_dir(parent_path = ""):
    current_date = datetime.datetime.now()
    dirname = current_date.strftime("%Y_%B_%d-%H_%M_%S")
    date = current_date.strftime("_%d")
    path = make_subfolder(dirname,parent_path)
    return path

def return_date():
    current_date = datetime.datetime.now()
    date = current_date.strftime("%Y_%B_%d")
    return date

def write_to_log(log_dir ="", log_entry = ""):
    with open(log_dir + "/log.txt", "a") as file:
        file.write(log_entry)

def send_as_mail(log_dir):
    log = log_dir + '/log.txt'
    conf_mat = log_dir + '/confusionMatrix.png'
    loss_and_acc = log_dir + '/loss_and_acc.png'
    contents = [ "Train sonuçları ve konfigürasyonu ekte yer almaktadır",
    log, loss_and_acc, conf_mat
    ]
    with yagmail.SMTP('viventedevelopment', 'yeniparrola2.1') as yag:
        yag.send('ademgunesen+viventedev@gmail.com', 'Train Sonuçları', contents)

def show_images(images: list, titles: list="Untitled    ", colorScale='gray', rows = 0, columns = 0) -> None:
    n: int = len(images)
    if rows == 0:
        rows=int(math.sqrt(n))
    if columns == 0:
        columns=(n//rows)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(rows, columns, i + 1)
        plt.imshow(images[i], cmap=colorScale)
        plt.title(titles[i])
    plt.show(block=True)

def get_computer_info():
    f = open('computer_info.json',)
    computer_info = json.load(f)
    print("Working on "+computer_info['name'])
    return computer_info

def initialize_metrics_csv(date):
    #initialize csv file with date and header
    header = ['log_dir', 'Model Name', 'Dense Layer', 'Dropout', 'Batch Size', 'Warmup Learning Rate', 'Learning Rate', 'Loss', 'Accuracy', 'ROC AUC', 'Best Threshold']#run once 
    with open(f'out/Train_Metrics_{date}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        csv_file = (f'out/Train_Metrics_{date}.csv')
    return csv_file 

def initialize_tasks_csv():
    #initialize csv file with date and header
    header = ['task_id', 'log_dir', 'task_state']#run once 
    with open(f'out/Tasks.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

def start_task(t_conf, g_conf, task_num):
    log_dir = make_log_dir("out/")                                                                                                                                                                                        
    t_conf.save(save_dir = log_dir)
    g_conf.save(save_dir = log_dir)
    start_info = [task_num, log_dir, 'Started']
    with open(f'out/Tasks.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(start_info)
    return log_dir

def stop_task(t_conf, task_num):
    start_info = [task_num, t_conf.log_dir, 'Completed']
    with open(f'out/Tasks.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(start_info)

def check_task_state(task_id):
    fields = []
    state = 'Pending'
    print(task_id)
    with open(f'out/Tasks.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            print(row[0],row[2])
            if (task_id==int(row[0])):
                state = row[2]
                print("Match!")
    return state

def save_metrics(csv_file, conf, loss, acc, roc_auc, best_thresh):
    #save same inputs and outputs to cvs file 
    row = []
    row = [conf.log_dir, conf.MODEL_NAME, conf.DENSE_LAYER, conf.DROP_OUT, conf.BATCH_SIZE, conf.WARMUP_LEARNING_RATE, conf.LEARNING_RATE, loss, acc, roc_auc, best_thresh]
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def create_to_csv(path):
    
    #creating the subset as validation by 0.2 (2.)
    df = pd.read_csv(path+'BadDataset/bad_dataset.csv')

    df_sub = df.sample(frac=0.1, random_state=1)#0.2, 2
    
    for index in df.index:
        for sub_index in df_sub.index:
            if index == sub_index:
                if df.iloc[index,2] == 'Training':
                    df.iloc[index,2] = 'Test'#'Validation'

    #df['Subsets'] = df_sub['Subsets'].replace({'Training':'Validation'})
    print(df['Subsets'])
    df.to_csv(path+'BadDataset/bad_dataset.csv', index=False)

def write_to_csv():
    #read images and labels from dataset and convert to csv file (1.)
    a = ['Images', 'Label', 'Subsets']#'Subsets'
    list = os.listdir('C:/Users/viven/Desktop/ROPProjects/Data/BadDataset/TestSetThumbnails/NotDisease')#Disease
    for i in range(len(list)):
        
        row = [list[i], '1', 'Training']#'0', 'Training'
        with open('bad_dataset.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            #writer.writerow(a)
            writer.writerow(row)  

def read_best_threshold(comp_info):
    #read best threshold from csv file for evaluate with test dataset
    file = comp_info["Codes"]+'/out/Train_Metrics.csv'
    df = pd.read_csv(comp_info["Codes"]+'/out/Train_Metrics.csv')
    with open(file) as f:
        reader = csv.reader(f, delimiter=",")
        for index in df.index:
            if df.iloc[index, 0] == 'out/2021_November_17-11_56_28/':
                best_thresh = df.iloc[index, 4]

    return best_thresh
