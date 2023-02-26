import re
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Normalization():
    """
    This class performs normalization on our entire dataset, as the confusion matrix is sparse without normalization.
    """
    _data_set = None
    _corpus = None
    _toxic_corpus = None
    _non_toxic_corpus =None
    
    def __init__(self, file_path = os.getcwd()+"/datasets/train.csv") -> None:
        """
        The constructor just tries to initialize the class attributes.
        Even those which are private.
        """
        try:
            
            self._data_set = pd.read_csv(file_path)
            
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
            self._data_set = self._data_set.drop(columns=["id", "obscene", "severe_toxic", "threat", "insult", "identity_hate"])
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

            for row in self._data_set.values:
                #row[0] is the comment_text made by the user, row[1] is the is_toxic flag.
                comment = self.normalize_comment(row[0])
                
                if len(comment) == 0 :
                    continue

                row[0] = comment
                row = (row[0], row[1])
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

    def __init__(self) -> None:
        """
        Constructor initializes the attributes of the class which are necessary.
        """
        pass

    def train_NB_model(self) -> BernoulliNB():
        """
        This method trains a BernoulliNB model, against the data set that we have and then returns a trained Model.
        """
        pass


obj = Normalization()
obj.normalize()
