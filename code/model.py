# imports
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

class model_dummy:
    def __init__(self):
        self.dummy_model = DummyClassifier(strategy="most_frequent")

    def Training(self, X_train, y_train):

        self.dummy_model.fit(X_train, y_train)
         
    def Prediction(self, X_test):

        self.pre = self.dummy_model.predict(X_test)
        return self.pre

    def accuracy(self, X_test, y_test):
        
        return self.dummy_model.score(y_test, X_test)
    
    def plot_confusion_matrix(self, cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                             label_=""):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Orginal label')
        plt.xlabel('Predicted label')
        plt.savefig(label_ + '.png', bbox_inches='tight')
    
    def Confusion(self, y_test, X_test, label_):

        cnf_matrix = confusion_matrix(y_test, X_test, labels=[0, 1, 2, 3, 4])
        self.plot_confusion_matrix(cnf_matrix, classes=["0","1","2","3","4"],normalize=False,
                      title= label_ + ' Confusion matrix, without normalization', label_=label_)
        