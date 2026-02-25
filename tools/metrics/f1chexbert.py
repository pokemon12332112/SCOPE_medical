import torch
import os
import logging
import numpy as np
import torch.nn as nn
import pandas as pd
import warnings

from collections import OrderedDict
from transformers import BertTokenizer
from transformers import BertModel, AutoModel, AutoConfig
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics._classification import _check_targets
from sklearn.utils.sparsefuncs import count_nonzero


warnings.filterwarnings("ignore")
logging.getLogger("urllib3").setLevel(logging.ERROR)


def download_model(repo_id, cache_dir, filename=None):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    if filename is not None:
        files = [filename]
    else:
        files = list(set(list_repo_files(repo_id=repo_id)).difference({'README.md', '.gitattributes'}))

    for f in files:
        try:
            hf_hub_download(repo_id=repo_id, filename=f, cache_dir=cache_dir, force_filename=f)
        except Exception as e:
            print(e)


def generate_attention_masks(batch, source_lengths, device):
    masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
    for idx, src_len in enumerate(source_lengths):
        masks[idx, src_len:] = 0
    return masks.to(device)


class bert_labeler(nn.Module):
    def __init__(self, p=0.1, freeze_embeddings=False, model_checkpoint=None, **kwargs):

        super(bert_labeler, self).__init__()

        config = AutoConfig.from_pretrained(model_checkpoint)
        self.bert = AutoModel.from_config(config)
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p)

        hidden_size = self.bert.pooler.dense.in_features
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, source_padded, attention_mask):

        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)
        out = []
        for i in range(14):
            out.append(self.linear_heads[i](cls_hidden))
        return out


def tokenize(impressions, tokenizer):
    imp = impressions.str.strip()
    imp = imp.replace('\n', ' ', regex=True)
    imp = imp.replace('\s+', ' ', regex=True)
    impressions = imp.str.strip()
    new_impressions = []
    for i in (range(impressions.shape[0])):
        tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
        if tokenized_imp:
            res = tokenizer.encode_plus(tokenized_imp)['input_ids']
            if len(res) > 512: 
                res = res[:511] + [tokenizer.sep_token_id]
            new_impressions.append(res)
        else:
            new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
    return new_impressions


class F1CheXbert(nn.Module):
    def __init__(self, refs_filename=None, hyps_filename=None, device=None, model_checkpoint=None,
                 chexbert_checkpoint=None, tokenizer_checkpoint=None, **kwargs):
        super(F1CheXbert, self).__init__()
        self.refs_filename = refs_filename
        self.hyps_filename = hyps_filename

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device(device)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_checkpoint)
        self.model = bert_labeler(model_checkpoint=model_checkpoint)

        state_dict = torch.load(chexbert_checkpoint, map_location=self.device)['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.target_names = [
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
            "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices", "No Finding"]

        self.target_names_5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
        self.target_names_5_index = np.where(np.isin(self.target_names, self.target_names_5))[0]

    def get_label(self, report, mode="rrg"):
        impressions = pd.Series([report])
        out = tokenize(impressions, self.tokenizer)
        batch = torch.LongTensor([o for o in out])
        src_len = [b.shape[0] for b in batch]
        attn_mask = generate_attention_masks(batch, src_len, self.device)
        out = self.model(batch.to(self.device), attn_mask)
        out = [out[j].argmax(dim=1).item() for j in range(len(out))]
        v = []
        if mode == "rrg":
            for c in out:
                if c == 0:
                    v.append('')
                if c == 3:
                    v.append(1)
                if c == 2:
                    v.append(0)
                if c == 1:
                    v.append(1)
            v = [1 if (isinstance(l, int) and l > 0) else 0 for l in v]

        elif mode == "classification":
            for c in out:
                if c == 0:
                    v.append('')
                if c == 3:
                    v.append(-1)
                if c == 2:
                    v.append(0)
                if c == 1:
                    v.append(1)
        else:
            raise NotImplementedError(mode)

        return v

    def forward(self, hyps, refs):
        if self.refs_filename is None:
            refs_chexbert = [self.get_label(l.strip()) for l in refs]
        else:
            if os.path.exists(self.refs_filename):
                refs_chexbert = [eval(l.strip()) for l in open(self.refs_filename).readlines()]
            else:
                refs_chexbert = [self.get_label(l.strip()) for l in refs]
                open(self.refs_filename, 'w').write('\n'.join(map(str, refs_chexbert)))

        hyps_chexbert = [self.get_label(l.strip()) for l in hyps]
        if self.hyps_filename is not None:
            open(self.hyps_filename, 'w').write('\n'.join(map(str, hyps_chexbert)))

        refs_chexbert_5 = [np.array(r)[self.target_names_5_index] for r in refs_chexbert]
        hyps_chexbert_5 = [np.array(h)[self.target_names_5_index] for h in hyps_chexbert]

        accuracy = accuracy_score(y_true=refs_chexbert_5, y_pred=hyps_chexbert_5)
        y_type, y_true, y_pred = _check_targets(refs_chexbert_5, hyps_chexbert_5)
        differing_labels = count_nonzero(y_true - y_pred, axis=1)
        pe_accuracy = (differing_labels == 0).astype(np.float32)

        cr = classification_report(refs_chexbert, hyps_chexbert, target_names=self.target_names, output_dict=True)
        cr_5 = classification_report(refs_chexbert_5, hyps_chexbert_5, target_names=self.target_names_5,
                                     output_dict=True)

        return accuracy, pe_accuracy, cr, cr_5

    def train(self, mode: bool = True):
        mode = False
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
