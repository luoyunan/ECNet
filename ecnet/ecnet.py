import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import scipy.stats
import pathlib
import copy
import time

from ecnet import vocab
from ecnet.model import LSTMPredictor
from ecnet.data import Dataset
from ecnet.utils import Saver, EarlyStopping, Logger



class ECNet(object):
    def __init__(self,
            output_dir=None,
            train_tsv=None, test_tsv=None,
            fasta=None, ccmpred_output=None,
            use_loc_feat=True, use_glob_feat=True,
            split_ratio=[0.9, 0.1],
            random_seed=42,
            nn_name='lstm', n_ensembles=1,
            d_embed=20, d_model=128, d_h=128, nlayers=1,
            batch_size=128, save_log=False):

        self.dataset = Dataset(
            train_tsv=train_tsv, test_tsv=test_tsv,
            fasta=fasta, ccmpred_output=ccmpred_output,
            use_loc_feat=use_loc_feat, use_glob_feat=use_glob_feat,
            split_ratio=split_ratio,
            random_seed=random_seed)
        self.saver = Saver(output_dir=output_dir)
        self.logger = Logger(logfile=self.saver.save_dir/'exp.log' if save_log else None)
        self.use_loc_feat = use_loc_feat
        self.use_glob_feat = use_glob_feat
        vocab_size = len(vocab.AMINO_ACIDS)
        seq_len = len(self.dataset.native_sequence)
        proj_loc_config = {
            'layer': nn.Linear,
            'd_in': seq_len + 1,
            'd_out': min(128, seq_len)
        }
        proj_glob_config = {
            'layer': nn.Identity,
            'd_in': 768,
            'd_out': 768,
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if nn_name in ['lstm', 'blstm']:
            self.models = [LSTMPredictor(
                d_embed=d_embed, d_model=d_model, d_h=d_h, nlayers=nlayers,
                vocab_size=vocab_size, seq_len=seq_len,
                bidirectional=True if nn_name == 'blstm' else False,
                use_loc_feat=use_loc_feat, use_glob_feat=use_glob_feat,
                proj_loc_config=proj_loc_config, proj_glob_config=proj_glob_config
            ).to(self.device) for _ in range(n_ensembles)]
        else:
            raise NotImplementedError

        self.criterion = F.mse_loss
        self.batch_size = batch_size
        self.optimizers = [optim.Adam(model.parameters()) for model in self.models]
        self._test_pack = None

    @property
    def test_pack(self):
        if self._test_pack is None:
            test_loader, test_df = self.dataset.get_dataloader(
                'test', batch_size=self.batch_size, return_df=True)
            self._test_pack = (test_loader, test_df)
        return self._test_pack

    @property
    def test_loader(self):
        return self.test_pack[0]

    @property
    def test_df(self):
        return self.test_pack[1]

    def load_pretrained_model(self, checkpoint_dir):
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            raise ValueError(f'{checkpoint_dir} is not a directory')
        for i in range(len(self.models)):
            checkpoint_path = f'{checkpoint_dir}/model_{i + 1}.pt'
            self.logger.info('Load pretrained model from {}'.format(checkpoint_path))
            pt = torch.load(checkpoint_path)
            model_dict = self.models[i].state_dict()
            model_pretrained_dict = {k: v for k, v in pt['model_state_dict'].items() if k in model_dict}
            model_dict.update(model_pretrained_dict)
            self.models[i].load_state_dict(model_dict)
            self.optimizers[i].load_state_dict(pt['optimizer_state_dict'])


    def load_single_pretrained_model(self, checkpoint_path, model=None, optimizer=None, is_resume=False):
        self.logger.info('Load pretrained model from {}'.format(checkpoint_path))
        pt = torch.load(checkpoint_path)
        model_dict = model.state_dict()
        model_pretrained_dict = {k: v for k, v in pt['model_state_dict'].items() if k in model_dict}
        model_dict.update(model_pretrained_dict)
        model.load_state_dict(model_dict)
        optimizer.load_state_dict(pt['optimizer_state_dict'])
        return (model, optimizer, pt['log_info']) if is_resume else (model, optimizer)


    def save_checkpoint(self, ckp_name=None, model_dict=None, opt_dict=None, log_info=None):
        ckp = {'model_state_dict': model_dict,
               'optimizer_state_dict': opt_dict}
        ckp['log_info'] = log_info
        self.saver.save_ckp(ckp, ckp_name)


    def train(self, epochs=1000, log_freq=100, eval_freq=50,
                patience=500, save_checkpoint=False, resume_path=None):
        assert eval_freq <= log_freq
        monitoring_score = 'corr'
        for midx, (model, optimizer) in enumerate(zip(self.models, self.optimizers), start=1):
            (train_loader, train_df), (valid_loader, valid_df) = \
                self.dataset.get_dataloader(
                    'train_valid', self.batch_size,
                    return_df=True, resample_train_valid=True)
            if resume_path is not None:
                model, optimizer, log_info = self.load_single_pretrained_model(
                    '{}/model_{}.pt'.format(resume_path, midx),
                    model=model, optimizer=optimizer, is_resume=True)
                start_epoch = log_info['epoch'] + 1
                best_score = log_info['best_{}'.format(monitoring_score)]
            else:
                start_epoch = 1
                best_score = None

            best_model_state_dict = None
            stopper = EarlyStopping(patience=patience, eval_freq=eval_freq, best_score=best_score)
            model.train()
            try:
                for epoch in range(start_epoch, epochs + 1):
                    time_start = time.time()
                    tot_loss = 0
                    for step, batch in tqdm(enumerate(train_loader, 1),
                        leave=False, desc=f'M-{midx} E-{epoch}', total=len(train_loader)):
                        y = batch['label'].to(self.device)
                        X = batch['seq_enc'].to(self.device)
                        if self.use_loc_feat:
                            loc_feat = batch['loc_feat'].to(self.device)
                        else:
                            loc_feat = None
                        if self.use_glob_feat:
                            glob_feat = batch['glob_feat'].to(self.device)
                        else:
                            glob_feat = None

                        optimizer.zero_grad()
                        output = model(X, glob_feat=glob_feat, loc_feat=loc_feat)
                        output = output.view(-1)
                        loss = self.criterion(output, y)

                        loss.backward()
                        optimizer.step()
                        tot_loss += loss.item()

                    if epoch % eval_freq == 0:
                        val_results = self.test(test_model=model, test_loader=valid_loader,
                            test_df=valid_df, mode='val')
                        model.train()
                        is_best = stopper.update(val_results['metric'][monitoring_score])
                        if is_best:
                            best_model_state_dict = copy.deepcopy(model.state_dict())
                            if save_checkpoint:
                                self.save_checkpoint(ckp_name='model_{}.pt'.format(midx),
                                    model_dict=model.state_dict(),
                                    opt_dict=optimizer.state_dict(),
                                    log_info={
                                        'epoch': epoch,
                                        'best_{}'.format(monitoring_score): stopper.best_score,
                                        'val_loss':val_results['loss'],
                                        'val_results':val_results['metric']
                                    })

                    if epoch % log_freq == 0:
                        train_results = self.test(test_model=model, test_loader=train_loader,
                                test_df=train_df, mode='val')
                        if (log_freq <= eval_freq) or (log_freq % eval_freq != 0):
                            val_results = self.test(test_model=model, test_loader=valid_loader,
                                test_df=valid_df, mode='val')
                        model.train()
                        self.logger.info(
                            'Model: {}/{}'.format(midx, len(self.models))
                            + '\tEpoch: {}/{}'.format(epoch, epochs)
                            + '\tTrain loss: {:.4f}'.format(tot_loss / step)
                            + '\tVal loss: {:.4f}'.format(val_results['loss'])                         
                            + '\t' + '\t'.join(['Val {}: {:.4f}'.format(k, v) \
                                    for (k, v) in val_results['metric'].items()])
                            + '\tBest {n}: {b:.4f}\t'.format(n=monitoring_score, b=stopper.best_score)
                            + '\t{:.1f} s/epoch'.format(time.time() - time_start)
                            )
                        time_start = time.time()

                    if stopper.early_stop:
                        self.logger.info('Eearly stop at epoch {}'.format(epoch))
                        break
            except KeyboardInterrupt:
                self.logger.info('Exiting model training from keyboard interrupt')
            if best_model_state_dict is not None:
                model.load_state_dict(best_model_state_dict)

            test_results = self.test(test_model=model, model_label='model_{}'.format(midx))
            test_res_msg = 'Testing Model {}: Loss: {:.4f}\t'.format(midx, test_results['loss'])
            test_res_msg += '\t'.join(['Test {}: {:.6f}'.format(k, v) \
                                for (k, v) in test_results['metric'].items()])
            self.logger.info(test_res_msg + '\n')


    def test(self, test_model=None, test_loader=None, test_df=None,
                checkpoint_dir=None, save_prediction=False,
                calc_metric=True, calc_loss=True, model_label=None, mode='test'):
        if checkpoint_dir is not None:
            self.load_pretrained_model(checkpoint_dir)
        if test_loader is None and test_df is None:
            test_loader = self.test_loader
            test_df = self.test_df
        test_models = self.models if test_model is None else [test_model]
        esb_ypred, esb_yprob = None, None
        esb_loss = 0
        for model in test_models:
            model.eval()
            y_true, y_pred, y_prob = None, None, None
            tot_loss = 0
            with torch.no_grad():
                for step, batch in tqdm(enumerate(test_loader, 1),
                        desc=mode, leave=False, total=len(test_loader)):
                    X = batch['seq_enc'].to(self.device)
                    if self.use_loc_feat:
                        loc_feat = batch['loc_feat'].to(self.device)
                    else:
                        loc_feat = None
                    if self.use_glob_feat:
                        glob_feat = batch['glob_feat'].to(self.device)
                    else:
                        glob_feat = None

                    output = model(X, glob_feat=glob_feat, loc_feat=loc_feat)
                    output = output.view(-1)
                    if calc_loss:
                        y = batch['label'].to(self.device)
                        loss = self.criterion(output, y)
                        tot_loss += loss.item()
                    y_pred = output if y_pred is None else torch.cat((y_pred, output), dim=0)

            y_pred = y_pred.detach().cpu() if self.device == torch.device('cuda') else y_pred.detach()
            esb_ypred = y_pred.view(-1, 1) if esb_ypred is None else torch.cat((esb_ypred, y_pred.view(-1, 1)), dim=1)
            esb_loss += tot_loss / step

        esb_ypred = esb_ypred.mean(axis=1).numpy()
        esb_loss /= len(test_models)

        if calc_metric:
            y_fitness = test_df['score'].values
            eval_results = scipy.stats.spearmanr(y_fitness, esb_ypred)[0]

        test_results = {}
        results_df = test_df.copy()
        results_df = results_df.drop(columns=['sequence'])
        results_df['prediction'] = esb_ypred
        test_results['df'] = results_df
        if save_prediction:
            self.saver.save_df(results_df, 'prediction.tsv')
        test_results['loss'] = esb_loss
        if calc_metric:
            test_results['metric'] = {'corr': eval_results}
        return test_results


if __name__ == "__main__":
    protein_name = 'MTH3_HAEAESTABILIZED_Tawfik2015'
    dataset_name = 'DeepSequence_Riesselman2018'
    ecnet = ECNet(
        output_dir='./tmp',
        train_tsv=f'../../output/mutagenesis/{dataset_name}/{protein_name}/data.tsv',
        fasta=f'../../output/mutagenesis/{dataset_name}/{protein_name}/native_sequence.fasta',
        ccmpred_output=f'../../output/homologous/{dataset_name}/{protein_name}/hhblits/ccmpred/{protein_name}.braw',
        split_ratio=[0.7, 0.1, 0.2],
        use_loc_feat=True, use_glob_feat=True,
        random_seed=42,
        nn_name='lstm', n_ensembles=3,
        d_embed=20, d_model=128, d_h=128,
        batch_size=128, save_log=False
    )
    ecnet.train(epochs=1000)
    test_results = ecnet.test(model_label='Test', mode='ensemble')
    test_res_msg = 'Testing Ensemble Model: Loss: {:.4f}\t'.format(test_results['loss'])
    test_res_msg += '\t'.join(['Test {}: {:.6f}'.format(k, v) for (k, v) in test_results['metric'].items()])
    ecnet.logger.info(test_res_msg + '\n')