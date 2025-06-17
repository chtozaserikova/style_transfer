import numpy as np

class EarlyStopper:
    def __init__(self, patience=30, min_delta=0):
        '''
        Arguments:
            patience (int): number of times to allow for no improvement before stopping the execution
            min_delta (float): minimum change counted as an improvement
        '''
        self.patience = patience 
        self.min_delta = min_delta
        self.counter = 0 # internal counter
        self.min_loss = np.inf

    # Return True when validation loss is not decreased by `min_delta` `patience` times 
    def early_stop(self, loss):
        if ((loss + self.min_delta) < self.min_loss):
            self.min_loss = loss
            self.counter = 0
        elif ((loss + self.min_delta) > self.min_loss):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def get_patience(self):
        return self.patience