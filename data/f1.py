import pandas as pd
from sklearn.metrics import f1_score
import sys

Y_TEST_PATH = 'y_test.csv'

y_test = pd.read_csv(Y_TEST_PATH)

def check_score(filepath):
    y_pred = pd.read_csv(filepath)
    assert 'wage' in y_pred.columns, "no header 'wage'"
    assert set(y_pred['wage']) == set([0,1]), 'predict label must be 0 or 1'
    assert len(y_pred) == len(y_test), f"your file has {len(y_pred)} rows, it must be {len(y_test)}"
    print(round(f1_score(y_test['wage'], y_pred['wage'], average='macro'), 4))

if __name__ == '__main__':
    args = sys.argv
    check_score(args[1])
    
## usage in Command Line (terminal)
## $ python3 f1.py PREDICTION_FILE_NAME
