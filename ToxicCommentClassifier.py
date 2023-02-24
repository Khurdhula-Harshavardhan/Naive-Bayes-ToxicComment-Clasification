import re
import os
import pandas as pd

class Normalization():
    """
    This class performs normalization on our entire dataset, as the confusion matrix is sparse without normalization.
    """
    _data_set = None
    _corpus = None
    def __init__(self, file_path = os.getcwd()+"/datasets/train.csv") -> None:
        """
        The constructor just tries to initialize the class attributes.
        Even those which are private.
        """
        try:
            
            self._data_set = pd.read_csv(file_path)
            print("[INFO] data_set has been initiallized")

            self._corpus = list()
        except Exception as e:
            print("[ERR] the following error occured while trying to open file %s: %s"%(file_path, str(e)))

    def normalize(self) -> None:
        """
        Normalization.normalize() is the method that performs normalization on the data.
        """
        try:
            pass
        except Exception as e:
            print("[ERR] the following error occured while trying to normalize the dataset! : %s"%(str(e)))

class BernoulliDistribution():
    def __init__(self) -> None:
        pass



obj = Normalization()
