import utils as util
import pandas as pd
from ae_helpers import generator, get_ae_model, plot_loss_and_acc
from configurators import train_config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

seed = 42
comp_info = util.get_computer_info()
x_train_path=comp_info["Datasets"]+'APTOS/dataset_15&19/'
x_valid_path=comp_info["Datasets"]+'APTOS/dataset_15&19/'
x_test_path =comp_info["Datasets"]+'APTOS/dataset_15&19/'


train_df = pd.read_csv(comp_info["Datasets"]+'APTOS/'+'ae_dataset.csv')
X_train = train_df[train_df['Subset'] == 'training']
X_valid = train_df[train_df['Subset'] == 'validation']

t_conf = train_config()
 
train_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_dataframe(
            dataframe   =X_train,
            directory   =x_train_path,
            x_col       ='Image', 
            target_size =(t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH),
            batch_size  =t_conf.BATCH_SIZE,
            class_mode = 'input',
            seed        =seed,
            shuffle     =True)

validation_generator = train_datagen.flow_from_dataframe(
            dataframe   =X_valid,
            directory   =x_train_path,
            x_col       ='Image', 
            target_size =(t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH),
            batch_size  =t_conf.BATCH_SIZE,
            class_mode = 'input',
            seed        =seed,
            shuffle     =True)

train_steps = len(X_train.index)//t_conf.BATCH_SIZE
val_steps = len(X_valid.index)//t_conf.BATCH_SIZE

ae_model = get_ae_model(t_conf)

es =EarlyStopping(monitor='val_loss', patience=5)
mc =ModelCheckpoint("out/mc_model",monitor="val_loss",save_best_only=True)

history=ae_model.fit(train_generator,
                epochs=100,
                steps_per_epoch=train_steps,
                batch_size=t_conf.BATCH_SIZE,
                shuffle=True,
                validation_steps=val_steps,
                validation_data = validation_generator,
                callbacks=[es,mc]
               )
#util.save_var(history,"hist")
#plot_loss_and_acc(history, save=False, saveDir='out/', fname='')
util.save_model(ae_model, "auotencoder1")