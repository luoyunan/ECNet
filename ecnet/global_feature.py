"""
Adapted from
https://github.com/songlab-cal/tape/blob/2e5bfa2249274392baf045df7de1490e92d47ed1/tape/datasets.py
https://github.com/songlab-cal/tape/blob/2e5bfa2249274392baf045df7de1490e92d47ed1/tape/training.py
"""
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from tape import utils
from tape.registry import registry
from tape.training import ForwardRunner
from tape.tokenizers import TAPETokenizer

logger = logging.getLogger(__name__)

class DataFrameDataset(Dataset):
    def __init__(self, seq_df):
        self._cache = []
        for seq_id, seq in zip(
            seq_df['ID'].values,
            seq_df['sequence'].values
        ):
            self._cache.append({
                'id': seq_id,
                'primary': seq,
                'protein_length': len(seq)
            })
        self._num_examples = len(self._cache)

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        return self._cache[index]

def _pad_sequences(sequences: Sequence[np.ndarray], constant_value=0) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
    array = np.zeros(shape, sequences[0].dtype) + constant_value

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

class EmbedDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 ):
        super().__init__()

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        if isinstance(data, pd.DataFrame):
            self.data = DataFrameDataset(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return item['id'], token_ids, input_mask

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        ids, tokens, input_mask = zip(*batch)
        ids = list(ids)
        tokens = torch.from_numpy(_pad_sequences(tokens))
        input_mask = torch.from_numpy(_pad_sequences(input_mask))
        return {'ids': ids, 'input_ids': tokens, 'input_mask': input_mask}  # type: ignore


class TAPEEncoder(object):
    def __init__(self,
        model_type: str = 'transformer',
        from_pretrained: str = None,
        batch_size: int = 128,
        model_config_file: Optional[str] = None,
        full_sequence_embed: bool = True,
        no_cuda: bool = False,
        seed: int = 42,
        tokenizer: str = 'iupac',
        num_workers: int = 4,
        log_level: Union[str, int] = logging.INFO,
        progress_bar: bool = True
    ):
        """
        Parameters
        ----------
        seq_df: pd.DataFrame object. Two columns are requried `ID` and `sequence`.
        """
        local_rank = -1  # TAPE does not support torch.distributed.launch for embedding
        device, n_gpu, is_master = utils.setup_distributed(local_rank, no_cuda)
        utils.setup_logging(local_rank, save_path=None, log_level=log_level)
        utils.set_random_seeds(seed, n_gpu)

        task_spec = registry.get_task_spec('embed')
        model = registry.get_task_model(
            model_type, task_spec.name, model_config_file, from_pretrained)
        model = model.to(device)
        runner = ForwardRunner(model, device, n_gpu)
        runner.initialize_distributed_model()
        runner.eval()
        self.local_rank = local_rank
        self.n_gpu = n_gpu
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.runner = runner
        self.full_sequence_embed = full_sequence_embed
        self.progress_bar = progress_bar

    def encode(self, sequences: [str]) -> np.ndarray:
        """
        Parameters
        ----------
        sequences: list of equal-length sequences

        Returns
        -------
        np array with shape (#sequences, length of sequences, embedding dim)
        """
        seq_df = pd.DataFrame({'ID': sequences, 'sequence': sequences})
        embed_dict = self.tape_embed(seq_df)
        encoding = np.array([embed_dict[s].numpy() for s in sequences])
        return encoding

    def tape_embed(self, seq_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        seq_df: pd.DataFrame with at least two columns `ID` and `sequence`

        Returns
        -------
        dict with ID as keys and embeddings as values. Embeddings are 
        pytorch tensor with shape (#sequences, length of sequences, embedding dim)
        """
        local_rank = self.local_rank
        n_gpu = self.n_gpu
        batch_size = self.batch_size
        num_workers = self.num_workers
        runner = self.runner
        full_sequence_embed = self.full_sequence_embed
        progress_bar = self.progress_bar

        dataset = EmbedDataset(seq_df)
        valid_loader = utils.setup_loader(dataset, batch_size, local_rank, n_gpu, 1, num_workers)

        embed_dict = {}
        with torch.no_grad():
            with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu):
                for batch in (tqdm(valid_loader, total=len(valid_loader),
                        desc='encode', leave=False) if progress_bar else valid_loader):
                    outputs = runner.forward(batch, no_loss=True)
                    ids = batch['ids']
                    sequence_embed = outputs[0]
                    pooled_embed = outputs[1]
                    sequence_lengths = batch['input_mask'].sum(1)
                    sequence_embed = sequence_embed.cpu()
                    sequence_lengths = sequence_lengths.cpu()

                    for seqembed, length, protein_id in zip(
                            sequence_embed, sequence_lengths, ids):
                        seqembed = seqembed[:length]
                        seqembed = seqembed[1:-1] # remove <cls> <sep>
                        if not full_sequence_embed:
                            seqembed = seqembed.mean(0)
                        embed_dict[protein_id] = seqembed
        return embed_dict

if __name__ == '__main__':
    df = pd.DataFrame([
        {"ID": '1', 'sequence': 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGEYAEWTYDDATKTFTQTE'},
        {"ID": '2', 'sequence': 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVGHEWTYDDATKTFTCTE'}
    ])
    tape_encoder = TAPEEncoder(progress_bar=False)
    encoding = tape_encoder.tape_embed(df)
    print(len(encoding))
    print(encoding['1'].shape)
    encoding = tape_encoder.encode(['SYT'])
    print(encoding.shape)