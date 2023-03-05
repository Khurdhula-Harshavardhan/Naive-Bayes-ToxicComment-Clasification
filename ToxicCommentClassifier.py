import re
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plotter
from sklearn.decomposition import PCA
import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class Normalization():
    """
    This class performs normalization on our entire dataset, as the confusion matrix is sparse without normalization.
    """
    __data_set = None
    _corpus = None #contains all comments. [(comment, is_toxic)]
    _toxic_corpus = None #contains all toxic comments. [(comment, is_toxic)]
    _non_toxic_corpus =None #contains all non toxic comments. [(comment, is_toxic)]
    
    def __init__(self, file_path) -> None:
        """
        The constructor just tries to initialize the class attributes.
        Even those which are private.
        """
        try:
            
            self.__data_set = pd.read_csv(file_path)
            
            print("[INFO] data_set has been initiallized")

            self._corpus = list()
            self._toxic_corpus =list()
            self._non_toxic_corpus = list()
            self._total_toxic_comments = 0
        except Exception as e:
            print("[ERR] the following error occured while trying to open file %s: %s"%(file_path, str(e)))

    def _get_total_toxic_comments_count(self) -> int:
        return len(self._toxic_corpus)
    
    def _get_total_non_toxic_comments_count(self) -> int:
        return len(self._non_toxic_corpus)
    
    def show_corpus_stats(self) -> None:
        """
        show_corpus_stats aims to perform repeatative prints for corpus metrics such as its length, etc.
        """
        try:
            if len(self._corpus) == 0:
                raise Exception("There is nothing within corpus!")
            else:
                print("\n" * 3)
                print("-------------------- CORPUS -----------------------")
                print("Total number of rows within CORPUS: " + str(len(self._corpus)))
                print("Corpus head: ")
                print(self._corpus[:5])
                print("Total number of Toxic comments within Corpus: %d"%(self._get_total_toxic_comments_count()))
                print("Total Number of Non - Toxic comments within Corpus: %d"%(self._get_total_non_toxic_comments_count()))
        except Exception as e:
            print("[ERR] The following error occured while fetching corpus metrics: %s"%(str(e)))

    def normalize(self) -> None:
        """
        Normalization.normalize() is the method that performs normalization on the data.
        """
        try:
            self.__data_set = self.__data_set.drop(columns=["id", "obscene", "severe_toxic", "threat", "insult", "identity_hate"])
            print("[INFO] Irrelevant columns have been dropped.")

            self.build_corpus() #creates a list of tuples, where each tuple is a combination of comment_text and is_toxic flag. :)
        except Exception as e:
            print("[ERR] the following error occured while trying to normalize the dataset! : %s"%(str(e)))

    def normalize_comment(self, comment) -> str:
        """
        normalizes a comment.
        """
        try:
            comment = comment.lower() #lower the text for consistency
            comment = re.sub("'", "",comment) #remove '  for words like: can't, didn't, it's, etc.
            comment = " ".join(re.findall("[a-z]+", comment)) #capture just words, excluding special characters, symbols and numbers.
            if len(comment.strip()) == 0 : #if the comment is empty, or a string of white space characters we discard it.
                    return "" #return an empty string.
            return comment #return the newly normalized string otherwise.
        except Exception as e:
            print("[ERR] the following error occured while trying to normalize a comment! : %s"%(str(e)))
    
    def build_corpus(self) -> None:
        """
        Normalize.build_corpus() aims at creating corpus which is a list of tuples :)
        """
        try:
            print("[PROCESS] Creating corpus from data_frame, this might take a while :)")

            for row in self.__data_set.values:
                #row[0] is the comment_text made by the user, row[1] is the is_toxic flag.
                comment = self.normalize_comment(row[0])
                
                if len(comment) == 0 :
                    continue

                row[0] = comment
                row = [row[0], row[1]]
                self._corpus.append(row)

                if row[1] == 1:
                    self._toxic_corpus.append(row)
                else:
                    self._non_toxic_corpus.append(row)
            
            print("[UPDATE] Corpus has been created successfully!")
        except Exception as e:
            print("[ERR] the following error occured while trying to create corpus out of data_frame. : %s"%(str(e)))

class BernoulliDistribution():
    """
    This is the main class that implements the required methods.
    1) train_NB_model(): which returns a trained model.
    2) test_NB_model(): which tests the given file against model that we create.
    """
    _normalizer = None
    __vectorizer = None
    __X = None
    __y = None
    __X_vectorized = None
    _X_train = None
    _X_test = None
    _y_train = None
    _y_test = None
    __model = None
    _accuracy = None
    _F1_score = None
    _recall = None
    _precision = None
    __MODEL_NAME = None #constant.
    def __init__(self) -> None:
        """
        Constructor initializes the attributes of the class which are necessary.
        """
        try:
            self.__vectorizer = CountVectorizer(binary=True)
            self.__y = list()
            self.__model = BernoulliNB()
            self._accuracy = float()
            self.__MODEL_NAME = "BernoulliClassifier.joblib"
            self._recall = float()
            self._precision = float()
            self._F1_score = float()
        except Exception as e:
            print("[ERR] The following error occured while trying to Initialize values for Bernoulli Class: "+str(e))

    def save_model(self) -> None:
        """
        Implements model persistence.
        """
        try:
            joblib.dump(self.__model, self.__MODEL_NAME)
            print("[INFO] The newly trained model has been saved as BernoulliClassifier.joblib")
        except Exception as e:
            print("[ERR] The following error occured while trying to dump model.")
    
    def load_model(self) -> bool:
        """
        Checks if the model has already been saved previously, to save time that is consumed to fucking train a model again.
        """
        try:
            print("[PROCESS] Trying to load model %s that might have been previously saved after fitting."%(self.__MODEL_NAME))
            self.__model = joblib.load(self.__MODEL_NAME)
            return True
        except Exception as e:
            #we shall return false here.
            print("[INFO] A previously trained model does not exist, hence creating one!")
            return False

    def _transform_data(self, data_to_be_transformed) -> list(list()):
        """
        Transforms the current set of statements into a sparse matrix.
        """
        try:
            return self.__vectorizer.fit_transform(data_to_be_transformed)
        except Exception as e:
            print("[ERR] The following error occured while trying to vectorize the train set: " +str(e))
    
    def check(self, ind) -> int():
        """
        This method tries to get the toxic label for the current vector.

        """
        try:
            val = self.__y[ind]
            return val
        except Exception as e:
            pass

    def visualize_data(self, comments) -> None:
        """
        This method plots a graph for all the unique documents within the dataset.
        """
        
        X = self.__vectorizer.fit_transform(comments).toarray()

        # perform PCA on the word counts
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plotter.figure(figsize=(8,8))
        # plot the comments on a scatter plot using PCA components 1 and 2
        plotter.scatter(X_pca[:,0], X_pca[:,1], c='r') #,c=['r' if  self.check(i) == 1 else 'b' for i in range(len(X_pca[:,0]))], cmap='coolwarm')
        plotter.xlabel('Feature Space')
        plotter.ylabel('Residual Variability')
        # plotter.axhline(y=X_pca[:, 1].mean(), color='r', linestyle='--', label="Mean RV")
        #plotter.axvline(x=X_pca[:, 0].mean(), color='gray', linestyle='--', label="Mean FS")
        #plotter.axhline(y=np.median(X_pca[:,1]), color='black', linestyle='--', label='Median')
        #plotter.axhline(y=np.mean(X_pca[:,1]), color='gray', linestyle='--', label='Mean')
        plotter.axhline(y=1, color='red', linestyle='--', label='RV Threshold')
        plotter.axvline(x =2, color='black', linestyle='--', label='SF Threshold')

        
        plotter.legend(loc='best')

        plotter.show()

    def train_NB_model(self, path_to_train_file =  "./datasets/train.csv") -> BernoulliNB():
        """
        This method trains a BernoulliNB model, against the data set that we have and then returns a trained Model.
        """
        try:
            print("[IMPORTANT] Preparing corpus please wait!")
            self._normalizer = Normalization(path_to_train_file) #create the object that should be useful for normalizing the entire dataset.
            self._normalizer.normalize() #Normalization is performed on entire dataset, and respective corpuses are created.
            # tox = pd.DataFrame(self._normalizer._toxic_corpus, columns=["comment", "is_toxic"])
            # perct = int(self._normalizer._get_total_non_toxic_comments_count() * 1)
            # print(perct)
            # non_tox = pd.DataFrame(self._normalizer._non_toxic_corpus[:perct], columns=["comment", "is_toxic"])
            self.__X = pd.DataFrame(self._normalizer._corpus, columns=["comment", "is_toxic"])
            
            self.__y = self.__X["is_toxic"]
            print("[PROCESS] Creating countVectors for the corpus please wait!")
            # toxic = list(self.__X["comment"].loc[self.__X["is_toxic"] == 1])
            # nonToxic = list(self.__X["comment"].loc[self.__X["is_toxic"] == 0])
            # balanced =   toxic[:25000] 
            # self.visualize_data(balanced)
            self.__X_vectorized  = self.__vectorizer.fit_transform(self.__X["comment"])
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self.__X_vectorized, self.__y, test_size=0.4, random_state=50)
            print("[INFO] Checking for a previosly stored JOB file...")
            if not self.load_model(): #if the model does not exist we must trian a new isntance of it and dump it.

                #train the model.
                print("[PROCESS] Fitting a BernoulliNaiveBayes, model to the data! Please wait this might take some time!")
                self.__model.fit(self._X_train, self._y_train) #train
                print("[UPDATE] The model has been trained successfully!")
                self.save_model() #model_persistence
                return self.__model
            else:
                print("[INFO] A previously trained model already exists, loading it.")
                return self.__model
            
        except Exception as e:
            print("[ERR] The following error occured while trying to Train a Bernoulli NB model: "+str(e))

    def traditional_test(self, model) -> float():
        """
        This method tests the trained model on split data and returns the accuracy score of the model.
        The split will be based on vectorizated Trained X as well.
        """
        try:
            predictions = model.predict(self._X_test)
            self._accuracy = accuracy_score(predictions, self._y_test)
            self._precision, self._recall, self._F1_score, _ = precision_recall_fscore_support(self._y_test, predictions, average='binary')
            print("-"*50)
            print("Here are the model metrics: ")
            print("[METRIC] Accuracy Score: %f"%(self._accuracy))
            print("[METRIC] Precision: %f"%(self._precision))
            print("[METRIC] F1-Score: %f"%(self._F1_score))
            print("[METRIC] Recall: %f"%(self._recall))
            
            return self._accuracy
        except Exception as e:
            print("[ERR] The following error occured while trying to test the model for accuracy! "+str(e))

    def transform_single_comment(self, comment) -> list(list()) :
        """
        This method transforms a comment passed as an arguement into a 2D sparse Vector.
        """
        try:
            return self.__vectorizer.transform(comment)
        except Exception as e:
            print("[ERR] The following error occured while trying to transform the test comment into a vector: " +str(e))

    def predict_if_toxic(self, comment) -> bool:
        """
        This method returns the prediction made by the model for a single comment, to check if it is toxic.
        """
        try:
            prediction  = self.__model.predict(self.transform_single_comment([comment])) #this is an array of predictions.
            if prediction[0] == 0:
                return "[PREDICTION] IsToxic: False"
            else:
                return "[PREDICTION] IsToxic: True" 
        except Exception as e:
            print("[ERR] The following error occured while trying to make a prediction: %s"%(str(e)))

    def test_NB_model(self, path_to_test_file, NB_model) -> None:
        try:
            self.__model = NB_model
            data_frame = pd.read_csv(path_to_test_file)
            data_frame = data_frame["comment_text"]

            predicted_data = list()
            print("[PROCESS] PREDICTING CLASSES FOR TESTSET THIS MIGHT TAKE A LONG WHILE!")
            for comment in data_frame.values:
                comment = self._normalizer.normalize_comment(comment)
                if len(comment.strip(" ")) == 0:
                    continue
                proba = self.__model.predict_proba(self.transform_single_comment([comment]))
                #print("0 class: %f, 1 class: %f"%(proba[0][0], proba[0][1]))
                predicted_class = None
                if proba[0][1]>0.5:
                    predicted_class = 1
                else:
                    predicted_class = 0
                val = [comment, float(proba[0][1]), predicted_class]
                predicted_data.append(val)
            
            output_data_frame= pd.DataFrame(predicted_data, columns=["comment_text", "toxic_probability","is_toxic"])
            print(output_data_frame.head())
            output_data_frame.to_csv("./datasets/_output.csv")
            print("[PROCESS] Successfully generated the output file!")
        except Exception as e:
            print("[ERR] the following error occured while trying to peform test: %s"%(str(e)))

