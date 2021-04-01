import sys
import logging
import torch
import pathlib
import numpy as np

class Logger(object):
    def __init__(self, logfile=None, level=logging.INFO):
        '''
        logfile: pathlib object
        '''
        self.logger = logging.getLogger()
        self.logger.setLevel(level)
        formatter = logging.Formatter("%(asctime)s\t%(message)s", "%Y-%m-%d %H:%M:%S")

        for hd in self.logger.handlers[:]:
            self.logger.removeHandler(hd)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        if logfile is not None:
            logfile.parent.mkdir(exist_ok=True, parents=True)
            fh = logging.FileHandler(logfile, 'w')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)


class Saver(object):
    def __init__(self, output_dir):        
        self.save_dir = pathlib.Path(output_dir)

    def save_ckp(self, pt, filename='checkpoint.pt'):
        self.save_dir.mkdir(exist_ok=True, parents=True)
        torch.save(pt, str(self.save_dir/filename))

    def save_df(self, df, filename):
        self.save_dir.mkdir(exist_ok=True, parents=True)
        df.to_csv(self.save_dir/filename, float_format='%.8f', index=False, sep='\t')


class EarlyStopping(object):
    def __init__(self, 
            patience=100, eval_freq=1, best_score=None, 
            delta=1e-9, higher_better=True):
        self.patience = patience
        self.eval_freq = eval_freq
        self.best_score = best_score
        self.delta = delta
        self.higher_better = higher_better
        self.counter = 0
        self.early_stop = False
    
    def not_improved(self, val_score):
        if np.isnan(val_score):
            return True
        if self.higher_better:
            return val_score < self.best_score + self.delta
        else:
            return val_score > self.best_score - self.delta
    
    def update(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            is_best = True
        elif self.not_improved(val_score):
            self.counter += self.eval_freq
            if (self.patience is not None) and (self.counter > self.patience):
                self.early_stop = True
            is_best = False
        else:
            self.best_score = val_score
            self.counter = 0
            is_best = True
        return is_best