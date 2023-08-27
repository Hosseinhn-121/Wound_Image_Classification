# imports
import os
import cv2
import pandas as pd
import numpy as np
from skimage import io
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from model import model_dummy
import scipy as sp
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from skimage import feature
from random_forest_model import model_RF

class Diagnosi:

    def __init__(self):
        print('Loading the data ...')
        self.pwat_db_template = pd.read_csv(os.getcwd() + '\\legendary-meme-main\\fonte\\risorse\\dati\\future_db.csv')
        print('Loading the data finished')
        self.rf_ac = []
        self.dummy_ac = []
    
    def Data_spliting(self):

        print('Spliting the data for training and testing ...')
        self.X_train_bordi, self.X_test_bordi, self.y_train_bordi, self.y_test_bordi = train_test_split(self.pwat_db_template.iloc[:,1:37], self.pwat_db_template["Bordi"], test_size=0.2)
        self.X_train_tipo_tessuto_necrotico, self.X_test_tipo_tessuto_necrotico, self.y_train_tipo_tessuto_necrotico, self.y_test_tipo_tessuto_necrotico = train_test_split(self.pwat_db_template.iloc[:,1:37], self.pwat_db_template["Tipo tessuto necrotico"], test_size=0.2)
        self.X_train_quantita_tessuto_necrotico, self.X_test_quantita_tessuto_necrotico, self.y_train_quantita_tessuto_necrotico, self.y_test_quantita_tessuto_necrotico = train_test_split(self.pwat_db_template.iloc[:,1:37], self.pwat_db_template["Quantita tessuto necrotico"], test_size=0.2)
        self.X_train_colore_cuta_perilesionale, self.X_test_colore_cuta_perilesionale, self.y_train_colore_cuta_perilesionale, self.y_test_colore_cuta_perilesionale = train_test_split(self.pwat_db_template.iloc[:,1:37], self.pwat_db_template["Colore cuta perilesionale"], test_size=0.2)
        self.X_train_tessuto_di_granulazione, self.X_test_tessuto_di_granulazione, self.y_train_tessuto_di_granulazione, self.y_test_tessuto_di_granulazione = train_test_split(self.pwat_db_template.iloc[:,1:37], self.pwat_db_template["Tessuto di granulazione"], test_size=0.2)
        print('Spliting the data for training and testing finished')
# dummy---------------------------------------------------------
    def compute_Prediction_dummy(self):
        # Prediction

        print('Predicting ...')
        self.bordi_predictions = self.dummy_bordi_model.Prediction(self.X_test_bordi)
        self.tipo_tessuto_necrotico_predictions = self.dummy_tipo_tessuto_necrotico_model.Prediction(self.X_test_tipo_tessuto_necrotico)
        self.quantita_tessuto_necrotico_predictions = self.dummy_quantita_tessuto_necrotico_model.Prediction(self.X_test_quantita_tessuto_necrotico)
        self.colore_cuta_perilesionale_predictions = self.dummy_colore_cuta_perilesionale_model.Prediction(self.X_test_colore_cuta_perilesionale)
        self.tessuto_di_granulazione_predictions = self.dummy_tessuto_di_granulazione_model.Prediction(self.X_test_tessuto_di_granulazione)
        print('Predicting finished\n')

        print('bordi_predictions: ', self.bordi_predictions)
        print('bordi_Y_test: ', list(self.y_test_bordi))
        print("\n")
        print('tipo_tessuto_necrotico_predictions: ', self.tipo_tessuto_necrotico_predictions)
        print('tipo_tessuto_necrotico_Y_test: ', list(self.y_test_tipo_tessuto_necrotico))
        print("\n")
        print('quantita_tessuto_necrotico_predictions: ', self.quantita_tessuto_necrotico_predictions)
        print('quantita_tessuto_necrotico_Y_test: ', list(self.y_test_quantita_tessuto_necrotico))
        print("\n")
        print('colore_cuta_perilesionale_predictions: ', self.colore_cuta_perilesionale_predictions)
        print('colore_cuta_perilesionale_Y_test: ', list(self.y_test_colore_cuta_perilesionale))
        print("\n")
        print('tessuto_di_granulazione_predictions: ', self.tessuto_di_granulazione_predictions)
        print('tessuto_di_granulazione_Y_test: ', list(self.y_test_tessuto_di_granulazione))
        print("\n")

    def compute_accuracy_dummy(self):
        #Accuracy

        print('Calculating the accuracy ...')
        bordi_accuracy = self.dummy_bordi_model.accuracy(self.y_test_bordi, self.bordi_predictions)
        tipo_tessuto_necrotico_accuracy =  self.dummy_tipo_tessuto_necrotico_model.accuracy(self.y_test_tipo_tessuto_necrotico, self.tipo_tessuto_necrotico_predictions)
        quantita_tessuto_necrotico_accuracy = self.dummy_quantita_tessuto_necrotico_model.accuracy(self.y_test_quantita_tessuto_necrotico, self.quantita_tessuto_necrotico_predictions)
        colore_cuta_perilesionale_accuracy = self.dummy_colore_cuta_perilesionale_model.accuracy(self.y_test_colore_cuta_perilesionale, self.colore_cuta_perilesionale_predictions)
        tessuto_di_granulazione_accuracy = self.dummy_tessuto_di_granulazione_model.accuracy(self.y_test_tessuto_di_granulazione, self.tessuto_di_granulazione_predictions)
        print('Calculating the accuracy finished\n')
        self.dummy_ac.append(bordi_accuracy)
        self.dummy_ac.append(tipo_tessuto_necrotico_accuracy)
        self.dummy_ac.append(quantita_tessuto_necrotico_accuracy)
        self.dummy_ac.append(colore_cuta_perilesionale_accuracy)
        self.dummy_ac.append(tessuto_di_granulazione_accuracy)

        print('bordi_accuracy: ', bordi_accuracy * 100)
        print('tipo_tessuto_necrotico_accuracy: ', tipo_tessuto_necrotico_accuracy * 100)
        print('quantita_tessuto_necrotico_accuracy: ', quantita_tessuto_necrotico_accuracy * 100)
        print('colore_cuta_perilesionale_accuracy: ', colore_cuta_perilesionale_accuracy * 100)
        print('tessuto_di_granulazione_accuracy: ', tessuto_di_granulazione_accuracy * 100)

    def plots_dummy(self):
        self.dummy_bordi_model.Confusion(self.y_test_bordi, self.bordi_predictions, "bordi")
        self.dummy_bordi_model.Confusion(self.y_test_tipo_tessuto_necrotico, self.tipo_tessuto_necrotico_predictions, "tipo_tessuto_necrotico")
        self.dummy_bordi_model.Confusion(self.y_test_quantita_tessuto_necrotico, self.quantita_tessuto_necrotico_predictions, "quantita_tessuto_necrotico")
        self.dummy_bordi_model.Confusion(self.y_test_colore_cuta_perilesionale, self.colore_cuta_perilesionale_predictions, "colore_cuta_perilesionale")
        self.dummy_bordi_model.Confusion(self.y_test_tessuto_di_granulazione, self.tessuto_di_granulazione_predictions, "tessuto_di_granulazione")

    def dummy_learn(self):
                # Creating and training the models
        # X is image and Y is the scores

        print('Training the models ...')

        # self.dummy_bordi_model = model_dummy()
        self.dummy_bordi_model = model_dummy()
        self.dummy_bordi_model.Training(self.X_train_bordi, self.y_train_bordi)

        # Tipo tessuto necrotico model
        self.dummy_tipo_tessuto_necrotico_model = model_dummy()
        self.dummy_tipo_tessuto_necrotico_model.Training(self.X_train_tipo_tessuto_necrotico, self.y_train_tipo_tessuto_necrotico)

        # Quantita tessuto necrotico model
        self.dummy_quantita_tessuto_necrotico_model = model_dummy()
        self.dummy_quantita_tessuto_necrotico_model.Training(self.X_train_quantita_tessuto_necrotico, self.y_train_quantita_tessuto_necrotico)

        # Colore cuta perilesionale model
        self.dummy_colore_cuta_perilesionale_model = model_dummy()
        self.dummy_colore_cuta_perilesionale_model.Training(self.X_train_colore_cuta_perilesionale, self.y_train_colore_cuta_perilesionale)

        # Tessuto di granulazione model
        self.dummy_tessuto_di_granulazione_model = model_dummy()
        self.dummy_tessuto_di_granulazione_model.Training(self.X_train_tessuto_di_granulazione, self.y_train_tessuto_di_granulazione)

        print('Training the models finished')
 
        # Prediction
        self.compute_Prediction_dummy()

        #Accuracy
        print('Calculating the accuracy ...')
        self.compute_accuracy_dummy()
        print('Calculating the accuracy finished\n')

        self.plots_dummy()
    
# end dummy------------------------------------ 

# random forest -------------------------------------------

    def compute_Prediction_RF(self):
        # Prediction

        print('Predicting RF ...')
        self.bordi_predictions = self.RF_bordi_model.Prediction(self.X_test_bordi)
        self.tipo_tessuto_necrotico_predictions = self.RF_tipo_tessuto_necrotico_model.Prediction(self.X_test_tipo_tessuto_necrotico)
        self.quantita_tessuto_necrotico_predictions = self.RF_quantita_tessuto_necrotico_model.Prediction(self.X_test_quantita_tessuto_necrotico)
        self.colore_cuta_perilesionale_predictions = self.RF_colore_cuta_perilesionale_model.Prediction(self.X_test_colore_cuta_perilesionale)
        self.tessuto_di_granulazione_predictions = self.RF_tessuto_di_granulazione_model.Prediction(self.X_test_tessuto_di_granulazione)
        print('Predicting RF finished\n')

        print('bordi_predictions: ', self.bordi_predictions)
        print('bordi_Y_test: ', list(self.y_test_bordi))
        print("\n")
        print('tipo_tessuto_necrotico_predictions: ', self.tipo_tessuto_necrotico_predictions)
        print('tipo_tessuto_necrotico_Y_test: ', list(self.y_test_tipo_tessuto_necrotico))
        print("\n")
        print('quantita_tessuto_necrotico_predictions: ', self.quantita_tessuto_necrotico_predictions)
        print('quantita_tessuto_necrotico_Y_test: ', list(self.y_test_quantita_tessuto_necrotico))
        print("\n")
        print('colore_cuta_perilesionale_predictions: ', self.colore_cuta_perilesionale_predictions)
        print('colore_cuta_perilesionale_Y_test: ', list(self.y_test_colore_cuta_perilesionale))
        print("\n")
        print('tessuto_di_granulazione_predictions: ', self.tessuto_di_granulazione_predictions)
        print('tessuto_di_granulazione_Y_test: ', list(self.y_test_tessuto_di_granulazione))
        print("\n")

    def compute_accuracy_RF(self):
        #Accuracy

        print('Calculating the RF accuracy ...')
        bordi_accuracy = self.RF_bordi_model.accuracy(self.X_test_bordi, self.y_test_bordi)
        tipo_tessuto_necrotico_accuracy =  self.RF_tipo_tessuto_necrotico_model.accuracy(self.X_test_tipo_tessuto_necrotico ,self.y_test_tipo_tessuto_necrotico)
        quantita_tessuto_necrotico_accuracy = self.RF_quantita_tessuto_necrotico_model.accuracy(self.X_test_quantita_tessuto_necrotico, self.y_test_quantita_tessuto_necrotico)
        colore_cuta_perilesionale_accuracy = self.RF_colore_cuta_perilesionale_model.accuracy(self.X_test_colore_cuta_perilesionale, self.y_test_colore_cuta_perilesionale)
        tessuto_di_granulazione_accuracy = self.RF_tessuto_di_granulazione_model.accuracy(self.X_test_tessuto_di_granulazione, self.y_test_tessuto_di_granulazione)
        print('Calculating the RF accuracy finished\n')

        self.rf_ac.append(bordi_accuracy)
        self.rf_ac.append(tipo_tessuto_necrotico_accuracy)
        self.rf_ac.append(quantita_tessuto_necrotico_accuracy)
        self.rf_ac.append(colore_cuta_perilesionale_accuracy)
        self.rf_ac.append(tessuto_di_granulazione_accuracy)

        print('bordi_accuracy: ', bordi_accuracy * 100)
        print('tipo_tessuto_necrotico_accuracy: ', tipo_tessuto_necrotico_accuracy * 100)
        print('quantita_tessuto_necrotico_accuracy: ', quantita_tessuto_necrotico_accuracy * 100)
        print('colore_cuta_perilesionale_accuracy: ', colore_cuta_perilesionale_accuracy * 100)
        print('tessuto_di_granulazione_accuracy: ', tessuto_di_granulazione_accuracy * 100)

    def plots_RF(self):
        self.RF_bordi_model.Confusion(self.y_test_bordi, self.bordi_predictions, "RF_bordi")
        self.RF_bordi_model.Confusion(self.y_test_tipo_tessuto_necrotico, self.tipo_tessuto_necrotico_predictions, "RF_tipo_tessuto_necrotico")
        self.RF_bordi_model.Confusion(self.y_test_quantita_tessuto_necrotico, self.quantita_tessuto_necrotico_predictions, "RF_quantita_tessuto_necrotico")
        self.RF_bordi_model.Confusion(self.y_test_colore_cuta_perilesionale, self.colore_cuta_perilesionale_predictions, "RF_colore_cuta_perilesionale")
        self.RF_bordi_model.Confusion(self.y_test_tessuto_di_granulazione, self.tessuto_di_granulazione_predictions, "RF_tessuto_di_granulazione")

    def RF_learn(self):
                # Creating and training the models
        # X is image and Y is the scores

        print('Training the RF models ...')

        self.RF_bordi_model = model_RF()
        self.RF_bordi_model.Training(self.X_train_bordi, self.y_train_bordi)

        # Tipo tessuto necrotico model
        self.RF_tipo_tessuto_necrotico_model = model_RF()
        self.RF_tipo_tessuto_necrotico_model.Training(self.X_train_tipo_tessuto_necrotico, self.y_train_tipo_tessuto_necrotico)

        # Quantita tessuto necrotico model
        self.RF_quantita_tessuto_necrotico_model = model_RF()
        self.RF_quantita_tessuto_necrotico_model.Training(self.X_train_quantita_tessuto_necrotico, self.y_train_quantita_tessuto_necrotico)

        # Colore cuta perilesionale model
        self.RF_colore_cuta_perilesionale_model = model_RF()
        self.RF_colore_cuta_perilesionale_model.Training(self.X_train_colore_cuta_perilesionale, self.y_train_colore_cuta_perilesionale)

        # Tessuto di granulazione model
        self.RF_tessuto_di_granulazione_model = model_RF()
        self.RF_tessuto_di_granulazione_model.Training(self.X_train_tessuto_di_granulazione, self.y_train_tessuto_di_granulazione)

        print('Training the RF models finished')
 
        # Prediction
        self.compute_Prediction_RF()

        #Accuracy
        print('Calculating the RF accuracy ...')
        self.compute_accuracy_RF()
        print('Calculating the RF accuracy finished\n')

        self.plots_RF()
    

# end random forest ---------------------------------------

    def chart(self):
        plt.figure(figsize=(15,8))
        x = ["","Bordi","Tipo tessuto necrotico","Quantita tessuto necrotico"
        	,"Colore cuta perilesionale","Tessuto di granulazione" ] 
        ax = plt.subplot(111)
        N = 5
        ind = np.arange(N)
        width = 0.2
        rects1 = ax.bar(ind, self.rf_ac, width, color=(0.1,0.2,0.5))
        print(rects1)
        rects2 = ax.bar(ind+width, self.dummy_ac, width, color=(0.6,0.1,0.4))
        ax.legend( (rects1[0], rects2[0]), ('RF', 'dummy') )
        ax.set_xticklabels(x)
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=4)
        plt.savefig('chart.png')

    def show_image(self, img, title):
        plt.figure()
        if title == "gray_border":
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(title)
        plt.savefig(title + '_image.png', bbox_inches='tight')

    def main_(self):

        self.Data_spliting() 
        print("start dummy model")
        self.dummy_learn()
        print("end dummy model")  
        print("start random forest model")
        self.RF_learn()
        print("end random forest model")    
        self.chart()   

aa = Diagnosi()
aa.main_()