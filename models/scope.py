import copy
import json
import math
import time
import torchmetrics
from typing import Dict
import torch
from torch import nn
import transformers
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers.configuration_utils import PretrainedConfig
from transformers import GPT2TokenizerFast, AutoModel, AutoConfig, AutoImageProcessor

from models.bert import Transformer, BertCrossLayer
from tools.resnet import ProjectionHead, get_extended_attention_mask
from tools.metrics.chexbert import RadGraphMetrics, F1CheXbertMetrics
from tools.metrics.coco import COCOCaptionMetrics
from tools.metrics.report_logger import ReportLogger
from tools.datasets import (MimiccxrPretrainDataset, MimiccxrFinetuneDataset,
                                             PretrainDinov2CollateFn, FinetuneDinov2CollateFn)


class Pretrain(pl.LightningModule):
    def __init__(
            self,
            args: Dict,
            tokenizer: GPT2TokenizerFast,
            logger,
            **kwargs,
    ):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.mylog = logger
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.prefetch_factor = 4
        self.val_min_losses = {
            "epoch": -1,
            "mpc_loss": 1000,
            "instance_loss": 1000,
            'loss': 1000
        }  

        self.train_loss_metric = {
            'loss': torchmetrics.MeanMetric().to(args['device']),
            'mpc_loss': torchmetrics.MeanMetric().to(args['device']),
            'instance_loss': torchmetrics.MeanMetric().to(args['device']),
        }
        self.val_loss_metric = {
            'loss': torchmetrics.MeanMetric().to(args['device']),
            'mpc_loss': torchmetrics.MeanMetric().to(args['device']),
            'instance_loss': torchmetrics.MeanMetric().to(args['device']),
        }
        self.test_loss_metric = {
            'loss': torchmetrics.MeanMetric(),
            'mpc_loss': torchmetrics.MeanMetric(),
            'instance_loss': torchmetrics.MeanMetric(),
        }

        self.image_processor = AutoImageProcessor.from_pretrained(args['rad_dino_path'])
        self.image_encoder = AutoModel.from_pretrained(args['rad_dino_path'])
        image_dim = self.image_encoder.config.hidden_size
        self.image_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.text_encoder = self.build_text_encoder()
        text_dim = self.text_encoder.config.hidden_size
        self.text_encoder.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = True

        self.image_projection = ProjectionHead(image_dim, args['hidden_size'] * 2, args['hidden_size'])
        self.text_projection = ProjectionHead(text_dim, args['hidden_size'] * 2, args['hidden_size'])

        self.ln_1 = nn.LayerNorm(image_dim)
        self.ln_2 = nn.LayerNorm(args['hidden_size'])

        self.vp2id = json.load(open(args['view_position_embed']))
        self.vp_pos_embed = nn.Parameter(torch.randn(len(self.vp2id), 1, image_dim), requires_grad=True)

        self.temp_pos_embed = nn.Parameter(torch.rand(3, 1, args['hidden_size']), requires_grad=True)

        self.fusion_multiview = Transformer(args['hidden_size'], args['multiview_fusion_num_layers'],
                                            heads=args['num_heads'],
                                            dim_head=args['hidden_size'] // 4,
                                            mlp_dim=args['hidden_size'])

        self.concepts = [
            "enlarged cardiomediastinum", "cardiomegaly", "lung opacity", "lung lesion",
            "edema", "consolidation", "pneumonia", "atelectasis",
            "pneumothorax", "pleural effusion", "pleural thickening",
            "fracture", "support devices"
        ]
        concept_token = self.tokenization(self.concepts, device='cpu')
        concept_emb = self.text_encoder(**concept_token, device='cuda')
        concept_proto = self.text_projection(concept_emb['last_hidden_state']).mean(dim=1, keepdim=True)
        self.prototypes = concept_proto
        

        self.mask_ratio = args['mask_ratio']
        self.decoder_dim = args['maskembed_decoder_dim']
        self.decoder_layers = args['maskembed_decoder_layers']
        self.decoder_nhead = args['maskembed_decoder_nhead']
        self.alpha = args['alpha']

        self.teacher = copy.deepcopy(self.image_encoder)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        decoder_layer = nn.TransformerEncoderLayer(d_model=self.decoder_dim,
                                                nhead=self.decoder_nhead,
                                                dim_feedforward=self.decoder_dim * 4,
                                                activation='gelu',
                                                batch_first=True)
        self.h_phi = nn.TransformerEncoder(decoder_layer, num_layers=self.decoder_layers)

        teacher_dim = args['hidden_size']
        self.decoder_to_teacher = nn.Linear(self.decoder_dim, teacher_dim)

    def locality_alignment_loss(self, images, view_positions, mask_ratio=None):
        device = images.device
        if mask_ratio is None:
            mask_ratio = getattr(self, "mask_ratio", 0.6)

        B = images.size(0)
        num_patches = 1370 

        patch_masks = self._random_mask(B, num_patches, mask_ratio, device)  
        masks = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device), patch_masks], dim=1)  

        with torch.no_grad():
            masked_images = self._apply_patch_mask(images, patch_masks)
            teacher_out = self.teacher(masked_images)
            teacher_embed = torch.cat([teacher_out["pooler_output"].unsqueeze(1),
                                    teacher_out["last_hidden_state"]], dim=1)  

            valid_view_positions = [vp.split("_")[0] for vp in view_positions]
            vp_embeds = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in valid_view_positions]
            vp_embeds = torch.cat(vp_embeds, dim=0)
            teacher_embed = self.ln_1(teacher_embed + vp_embeds)

        student_out = self.image_encoder(images)
        student_embed = torch.cat([student_out["pooler_output"].unsqueeze(1),
                                student_out["last_hidden_state"]], dim=1)  

        vp_embeds = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in valid_view_positions]
        vp_embeds = torch.cat(vp_embeds, dim=0)
        student_embed = self.ln_1(student_embed + vp_embeds)

        masked_student_embed = student_embed.clone()
        masked_student_embed[masks] = 0.0

        decoded = self.h_phi(masked_student_embed)
        decoded = self.decoder_to_teacher(decoded)  

        mask_float = masks.unsqueeze(-1).to(decoded.dtype)
        diff = (decoded - teacher_embed) * mask_float
        denom = mask_float.sum()
        loss = self.alpha * diff.pow(2).sum() / (denom + 1e-6)

        return loss

    def _random_mask(self, batch_size, num_patches, mask_ratio, device):
        num_mask = int(mask_ratio * num_patches)
        masks = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        for b in range(batch_size):
            perm = torch.randperm(num_patches, device=device)
            masks[b, perm[:num_mask]] = True
        return masks


    def _apply_patch_mask(self, images, patch_masks):
        B, C, H, W = images.shape
        num_patches = patch_masks.size(1)  

        grid_cols = int(num_patches ** 0.5)
        grid_rows = (num_patches + grid_cols - 1) // grid_cols  
        patch_h, patch_w = H // grid_rows, W // grid_cols

        masked_images = images.clone()
        for b in range(B):
            mask = patch_masks[b]
            if mask.numel() < grid_rows * grid_cols:
                pad_size = grid_rows * grid_cols - mask.numel()
                mask = torch.cat([mask, torch.zeros(pad_size, dtype=torch.bool, device=mask.device)])
            mask = mask.view(grid_rows, grid_cols)
            for i in range(grid_rows):
                for j in range(grid_cols):
                    if mask[i, j]:
                        h0, h1 = i * patch_h, min((i + 1) * patch_h, H)
                        w0, w1 = j * patch_w, min((j + 1) * patch_w, W)
                        masked_images[b, :, h0:h1, w0:w1] = 0.0
        return masked_images

    def build_text_encoder(self):
        enc_config = AutoConfig.from_pretrained(self.args['cxr_bert_path'], trust_remote_code=True)
        enc_config.vocab_size = len(self.tokenizer)
        enc_config.eos_token_id = self.tokenizer.eos_token_id
        enc_config.bos_token_id = self.tokenizer.bos_token_id
        enc_config.pad_token_id = self.tokenizer.pad_token_id
        enc_config.num_hidden_layers = self.args['text_encoder_num_layers']
        enc_config.max_length = 200
        return AutoModel.from_pretrained(
            self.args['cxr_bert_path'],
            config=enc_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = MimiccxrPretrainDataset(self.args, 'train', self.tokenizer)
            self.val_set = MimiccxrPretrainDataset(self.args, 'val', self.tokenizer)
            print(
                "No. of training & validation examples: {} & {}.".format(
                    self.train_set.__len__(), self.val_set.__len__()
                )
            )
            self.mylog.info("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:  
            self.test_set = MimiccxrPretrainDataset(self.args, 'test', self.tokenizer)
            print("No. of test examples: {}.".format(self.test_set.__len__()))
            self.mylog.info("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        collate_fn = PretrainDinov2CollateFn(self.args['images_dir'], self.image_processor,
                                             self.args['is_multiview_learning'], self.args['is_prior_scan'])
        return DataLoader(
            self.train_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        collate_fn = PretrainDinov2CollateFn(self.args['images_dir'], self.image_processor,
                                             self.args['is_multiview_learning'], self.args['is_prior_scan'])
        return DataLoader(
            self.val_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        collate_fn = PretrainDinov2CollateFn(self.args['images_dir'], self.image_processor,
                                             self.args['is_multiview_learning'], self.args['is_prior_scan'])
        return DataLoader(
            self.test_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn
        )

    def configure_optimizers(self):
        if self.args['task'] == 'pretrain':
            optimiser = torch.optim.AdamW(self.parameters(), lr=self.args['pt_lr'])
            lr_scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.1, patience=5)
            return {
                "optimizer": optimiser,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': self.args['monitor_metric'],
                    'frequency': 1
                }
            }
        else:
            pretrain_main_params, finetune_main_params = [], []
            if self.args['load'] is not None:
                checkpoint = torch.load(self.args['load'])['state_dict']
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    if name in checkpoint:
                        pretrain_main_params.append(param)
                    else:
                        finetune_main_params.append(param)
            else:
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    finetune_main_params.append(param)

            optimiser = torch.optim.AdamW(
                [{'params': pretrain_main_params, 'lr': self.args['pt_lr']},
                 {'params': finetune_main_params, 'lr': self.args['ft_lr']}])

            lr_scheduler = ReduceLROnPlateau(optimiser, mode='max', factor=0.1, patience=5)
            return {
                "optimizer": optimiser,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': self.args['monitor_metric'],
                    'frequency': 1   
                }
            }

    def tokenization(self, text, pair_text=None, device=None):
        if pair_text is None:
            inputs = self.tokenizer(text, padding=True, return_tensors='pt', return_token_type_ids=True,
                                    max_length=self.args['max_length'], truncation=True)
        else:
            inputs = self.tokenizer(text, pair_text, padding=True, return_token_type_ids=True,
                                    return_tensors='pt', max_length=200, truncation=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        return inputs

    def multiple_positive_contrastive_learning(self, global_image_embed, patient_ids, view_positions):
        valid_images_id = [i for i, vp in enumerate(view_positions) if 'prior' not in vp]
        valid_num_images = len(valid_images_id)
        patient_ids = patient_ids[:valid_num_images]
        global_image_embed = global_image_embed[:valid_num_images]

        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(float)
        labels = torch.from_numpy(labels).to(global_image_embed)
        labels.fill_diagonal_(0.0)

        idx = torch.argwhere(labels.sum(1) != 0).reshape(-1)
        if len(idx) == 0:
            return torch.tensor([0.0], requires_grad=True, device=global_image_embed.device)
        global_image_embed, labels = global_image_embed[idx], labels[idx][:, idx]
        labels = labels / labels.sum(1, keepdim=True)

        global_image_embed = F.normalize(global_image_embed, dim=-1, p=2)
        logits = global_image_embed @ global_image_embed.T / self.args['temp']
        logits.fill_diagonal_(-1e9)

        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - logits_max.detach()
        loss = F.cross_entropy(logits, labels)
        return loss

    def multiview_fusion_network(self, image_embed, patient_ids, batch_size, view_positions):
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(int)
        labels = torch.from_numpy(labels)
        labels.fill_diagonal_(0)

        new_image_embed = []
        for i in range(batch_size):
            if labels[i].sum() == 0:
                new_image_embed.append(image_embed[i])
                continue
            multiview_image_embed = torch.cat([image_embed[j] for j, tag in enumerate(labels[i]) if tag == 1], dim=0)
            cur_image_embed = self.fusion_multiview(image_embed[i], multiview_image_embed,
                                                    multiview_image_embed)

            new_image_embed.append(cur_image_embed)
        new_image_embed = torch.stack(new_image_embed, dim=0)
        return new_image_embed


    def local_text_token_alignment_loss(self, local_image_embed, local_text_embed):

        t_att_sim = local_text_embed @ local_image_embed.permute(0, 2, 1).contiguous()
        t_att_sco = F.softmax(t_att_sim / math.sqrt(local_image_embed.shape[2]), dim=-1)
        t_att_output = torch.bmm(t_att_sco, local_image_embed)

        device = local_image_embed.device

        t_att_output = F.normalize(t_att_output, dim=-1, p=2)
        local_text_embed = F.normalize(local_text_embed, dim=-1, p=2)

        word_sim = torch.bmm(local_text_embed, t_att_output.permute(0, 2, 1).contiguous()) / self.args['region_temp']
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
        word_targets = torch.arange(word_sim.shape[1]).long().repeat(word_sim.shape[0]).to(device)
        loss_word_1 = F.cross_entropy(word_sim_1, word_targets)

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = F.cross_entropy(word_sim_2, word_targets)
        loss_word = (loss_word_2 + loss_word_1) / 2.0
        return loss_word

    def encoder_forward(self, images, inputs, view_positions):
        outputs = self.image_encoder(images)
        image_embed = torch.cat([outputs['pooler_output'].unsqueeze(dim=1), outputs['last_hidden_state']], dim=1)

        valid_view_positions = [vp.split('_')[0] for vp in view_positions]
        image_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in valid_view_positions]

        image_embed = torch.cat(image_pos_embed, dim=0) + image_embed
        image_embed = self.ln_1(image_embed)

        image_embed = self.image_projection(image_embed)  

        text_embed = self.text_encoder(**inputs)
        text_embed = self.text_projection(text_embed['last_hidden_state'])  

        return image_embed, text_embed

    def prototypes_loss(self, i_ft, t_ft, tmp=0.07, lambda_commit=0.25):
        i_ft = F.normalize(i_ft, dim=-1)
        t_ft = F.normalize(t_ft, dim=-1)
        C = F.normalize(self.prototypes.squeeze(1), dim=-1)

        I2P = i_ft @ C.T
        T2P = t_ft @ C.T

        Tg = sinkhorn_transport(1 - I2P)
        Tf = sinkhorn_transport(1 - T2P)

        L_align = F.mse_loss(Tg, Tf)

        i_q = Tg @ C
        t_q = Tf @ C
        L_commit = F.mse_loss(i_ft, i_q) + F.mse_loss(t_ft, t_q)

        sim = (i_q @ t_q.T) / tmp
        labels = torch.arange(sim.size(0), device=sim.device)
        L_contrast = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))

        L_total = L_align + lambda_commit * L_commit + L_contrast

        return L_total, Tg, Tf
        

    def forward(self, images, reports, patient_ids, view_positions):
        device = images.device
        report_inputs = self.tokenization(reports, device=device)
        batch_size = len(reports)
        image_embed, text_embed = self.encoder_forward(images, report_inputs, view_positions)
        if self.args['using_maskembed']:
            loc_loss = self.locality_alignment_loss(images, view_positions)
        else:
            loc_loss = 0

        mul_pos_loss = torch.tensor([0.0])
        if self.args['using_mpc_loss']:
            mul_pos_loss = self.multiple_positive_contrastive_learning(image_embed[:, 0, :],
                                                                       patient_ids, view_positions)
        temporal_pos_embed = []
        for vp in view_positions:
            if 'prior' not in vp:
                temporal_pos_embed.append(self.temp_pos_embed[0].unsqueeze(0))
            else:
                if 'latest' in vp:
                    temporal_pos_embed.append(self.temp_pos_embed[1].unsqueeze(0))
                else:
                    temporal_pos_embed.append(self.temp_pos_embed[2].unsqueeze(0))
        image_embed = image_embed + torch.cat(temporal_pos_embed, dim=0)
        image_embed = self.ln_2(image_embed)

        if self.args['is_multiview_learning']:
            image_embed = self.multiview_fusion_network(image_embed, patient_ids, batch_size, view_positions)
        else:
            image_embed = image_embed[:batch_size]


        self.prototypes = self.prototypes.to(device)
        instance_loss = self.prototypes_loss(image_embed[:, 0, :], text_embed[:, 0, :])[0]
        if self.args['using_local_loss']:
            sen_text_loss = self.local_text_token_alignment_loss(image_embed[:, 1:, :], text_embed[:, 1:, :])
            if self.args['using_mpc_loss']:
                return {
                    'sen_text_loss': sen_text_loss,
                    'instance_loss': instance_loss,
                    'mpc_loss': mul_pos_loss,
                    'loss': instance_loss + sen_text_loss + mul_pos_loss + loc_loss
                }
            else:
                return {
                    'sen_text_loss': sen_text_loss,
                    'instance_loss': instance_loss,
                    'loss': instance_loss + sen_text_loss + loc_loss
                }
        else:
            if self.args['using_mpc_loss']:
                return {
                    'instance_loss': instance_loss,
                    'mpc_loss': mul_pos_loss,
                    'loss': instance_loss + mul_pos_loss + loc_loss
                }
            else:
                return {
                    'instance_loss': instance_loss,
                    'loss': instance_loss + loc_loss
                }

    def training_step(self, batch, batch_idx):
        image_ids, images, reports, patient_ids, view_positions = batch

        loss_dict = self(images, reports, patient_ids, view_positions)

        self.log_dict({f'train_step_{k}': v for k, v in loss_dict.items()}, on_step=True, on_epoch=False,
                      batch_size=len(reports),
                      prog_bar=True, sync_dist=True)
        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().item(), 2)}" for k, v in loss_dict.items()])
            self.mylog.info(
                f"Epoch {self.current_epoch}, training step {batch_idx}/{self.trainer.num_training_batches}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")

        for key, loss in loss_dict.items():
            if f"{key}" in self.train_loss_metric:
                self.train_loss_metric[f"{key}"].update(loss.detach())

        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        image_ids, images, reports, patient_ids, view_positions = batch

        loss_dict = self(images, reports, patient_ids, view_positions)

        self.log_dict({f'val_step_{k}': v for k, v in loss_dict.items()}, on_epoch=False, on_step=True,
                      batch_size=len(reports),
                      prog_bar=True, sync_dist=True)

        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().item(), 2)}" for k, v in loss_dict.items()])
            self.mylog.info(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")

        for key, loss in loss_dict.items():
            if f"{key}" in self.val_loss_metric:
                self.val_loss_metric[f"{key}"].update(loss)

    def test_step(self, batch, batch_idx):
        image_ids, images, reports, patient_ids, view_positions = batch
        loss_dict = self(images, reports, patient_ids, view_positions)

        self.log_dict({f'test_step_{k}': v for k, v in loss_dict.items()}, on_epoch=False, on_step=True,
                      batch_size=len(reports),
                      prog_bar=True, sync_dist=True)
        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().item(), 2)}" for k, v in loss_dict.items()])
            self.mylog.info(f"Epoch {self.current_epoch}, testing step {batch_idx}/{self.trainer.num_test_batches[0]}, "
                            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")
        for key, loss in loss_dict.items():
            if f"{key}" in self.test_loss_metric:
                self.test_loss_metric[f"{key}"].update(loss)

    def on_train_epoch_end(self):
        cur_all_loss = {}
        for key, metric in self.train_loss_metric.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[key] = avg_metric
        self.log_dict({f'train_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True,
                      on_step=False, prog_bar=True)

        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 2)}" for k, v in cur_all_loss.items()])
        self.mylog.info(
            f"Epoch {self.current_epoch}, Training is over, "
            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}"
            "\n###############################################################"
        )

    def on_validation_epoch_end(self):
        cur_all_loss = {}
        for key, metric in self.val_loss_metric.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[key] = avg_metric
        self.log_dict({f'val_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True, on_step=False, prog_bar=True)

        if cur_all_loss['loss'] < self.val_min_losses["loss"]:
            self.val_min_losses = {**cur_all_loss, "epoch": self.current_epoch}

        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 2)}" for k, v in cur_all_loss.items()])
        best_loss_item = ', '.join([f"{k} = {v}" for k, v in self.val_min_losses.items()])
        self.mylog.info(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, Validation is over, current val loss:"
            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}\n"
            f"best validation loss: {best_loss_item}\n"
        )

    def on_test_epoch_end(self):
        cur_all_loss = {}
        for key, metric in self.test_loss_metric.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[key] = avg_metric
        self.log_dict({f'test_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True, on_step=False, prog_bar=True)

        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 2)}" for k, v in cur_all_loss.items()])
        self.mylog.info(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, test is over, current loss:"
            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}\n"
        )


class Finetune(pl.LightningModule):
    def __init__(
            self,
            args: Dict,
            tokenizer: GPT2TokenizerFast,
            logger,
            **kwargs,
    ):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.mylog = logger
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.prefetch_factor = None
        self.val_best_scores = {
            "best_epoch": -1,
            "best_monitor_metric": -1.0,
        }
        self.time_sum = 0

        self.train_loss_metric = torchmetrics.MeanMetric()

        self.val_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"])
        self.test_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"], save=False)

        self.val_f1chexbert_metrics = F1CheXbertMetrics(
            chexbert_path=args['chexbert_path'],
            model_path=args['bert_path'],
            mbatch_size=16,
            exp_dir=args['exp_dir_trial'],
        )
        self.test_f1chexbert_metrics = F1CheXbertMetrics(
            chexbert_path=args['chexbert_path'],
            model_path=args['bert_path'],
            mbatch_size=16,
            exp_dir=args['exp_dir_trial'],
        )

        self.val_radgraph_metrics = RadGraphMetrics(
            radgraph_path=self.args['radgraph_path'],
            mbatch_size=16,
            exp_dir=args['exp_dir_trial'],
        )
        self.test_radgraph_metrics = RadGraphMetrics(
            radgraph_path=self.args['radgraph_path'],
            mbatch_size=16,
            exp_dir=args['exp_dir_trial'],
        )


        self.val_report_logger = ReportLogger(exp_dir=args['exp_dir_trial'], split='val_reports')
        self.test_report_logger = ReportLogger(exp_dir=args['exp_dir_trial'], split='test_reports')


        self.image_processor = AutoImageProcessor.from_pretrained(args['rad_dino_path'])
        self.image_encoder = AutoModel.from_pretrained(args['rad_dino_path'])
        self.image_encoder.eval()
        image_dim = self.image_encoder.config.hidden_size
        for param in self.image_encoder.parameters():
            param.requires_grad = False


        self.text_encoder = self.build_text_encoder()
        text_dim = self.text_encoder.config.hidden_size
        self.text_encoder.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = True


        self.image_projection = ProjectionHead(image_dim, args['hidden_size'] * 2, args['hidden_size'])
        self.text_projection = ProjectionHead(text_dim, args['hidden_size'] * 2, args['hidden_size'])


        self.ln_1 = nn.LayerNorm(image_dim)
        self.ln_2 = nn.LayerNorm(args['hidden_size'])
        self.ln_3 = nn.LayerNorm(args['hidden_size'])


        self.vp2id = json.load(open(args['view_position_embed']))
        self.vp_pos_embed = nn.Parameter(torch.randn(len(self.vp2id), 1, image_dim), requires_grad=True)

        self.temp_pos_embed = nn.Parameter(torch.rand(3, 1, args['hidden_size']), requires_grad=True)

        self.type_pos_embed = nn.Parameter(torch.rand(2, 1, args['hidden_size']), requires_grad=True)

        self.fusion_multiview = Transformer(args['hidden_size'], args['multiview_fusion_num_layers'],
                                            heads=args['num_heads'],
                                            dim_head=args['hidden_size'] // 4,
                                            mlp_dim=args['hidden_size'])

 
        self.text_decoder = self.build_text_decoder()

 
        fusion_multimodal_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args['bert_path'],
            vocab_size=len(self.tokenizer),
            hidden_size=args["hidden_size"],
            num_hidden_layers=args["cross_modal_fusion_num_layers"],
            num_attention_heads=args["num_heads"],
            max_position_embeddings=512,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        self.fusion_multimodal = nn.ModuleList(
            [BertCrossLayer(fusion_multimodal_config) for _ in range(args['cross_modal_fusion_num_layers'])])

    def build_text_encoder(self):
        enc_config = AutoConfig.from_pretrained(self.args['cxr_bert_path'], trust_remote_code=True)
        enc_config.vocab_size = len(self.tokenizer)
        enc_config.eos_token_id = self.tokenizer.eos_token_id
        enc_config.bos_token_id = self.tokenizer.bos_token_id
        enc_config.pad_token_id = self.tokenizer.pad_token_id
        enc_config.num_hidden_layers = self.args['text_encoder_num_layers']
        enc_config.max_length = 200
        return AutoModel.from_pretrained(
            self.args['cxr_bert_path'],
            config=enc_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True)

    def build_text_decoder(self):
        config = transformers.GPT2Config.from_pretrained(self.args['distilgpt2_path'])
        config.add_cross_attention = True
        config.is_decoder = True
        config.vocab_size = len(self.tokenizer)
        if self.args['cvt2distilgpt2_path'] is None:
            decoder = transformers.GPT2LMHeadModel.from_pretrained(
                self.args['distilgpt2_path'],
                config=config,
                ignore_mismatched_sizes=True
            )
            decoder.resize_token_embeddings(len(self.tokenizer))
        else:
            decoder = transformers.GPT2LMHeadModel(config=config)
            decoder.resize_token_embeddings(len(self.tokenizer))

            checkpoint = torch.load(self.args['cvt2distilgpt2_path'])['state_dict']
            checkpoint = {k.split('decoder.encoder_decoder.decoder.')[-1]: v for k, v in checkpoint.items() if
                          'decoder' in k}
            curr_state_dict = decoder.state_dict()
            valid_state_dict = {k: v for k, v in checkpoint.items() if
                                k in curr_state_dict and v.shape == curr_state_dict[k].shape}
            curr_state_dict.update(valid_state_dict)
            decoder.load_state_dict(curr_state_dict)

        class DummyEncoder:
            main_input_name = 'dummy'

            class DummyConfig(PretrainedConfig):
                model_type = 'bert'

            config = DummyConfig()

            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size

            def forward(self, *args, **kwargs):
                pass

            def get_output_embeddings(cls):
                return None

        dummy_encoder = DummyEncoder(hidden_size=decoder.config.hidden_size)

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_decoder = transformers.EncoderDecoderModel(encoder=dummy_encoder, decoder=decoder)

        return Decoder()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:

            self.train_set = MimiccxrFinetuneDataset(self.args, 'train', self.tokenizer)
            self.val_set = MimiccxrFinetuneDataset(self.args, 'val', self.tokenizer)
            print(
                "No. of training & validation examples: {} & {}.".format(
                    self.train_set.__len__(), self.val_set.__len__()
                )
            )
            self.mylog.info("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:  
            self.test_set = MimiccxrFinetuneDataset(self.args, 'test', self.tokenizer)
            print("No. of test examples: {}.".format(self.test_set.__len__()))
            self.mylog.info("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        collate_fn = FinetuneDinov2CollateFn(self.args, self.image_processor)
        return DataLoader(
            self.train_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        collate_fn = FinetuneDinov2CollateFn(self.args, self.image_processor)
        return DataLoader(
            self.val_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        collate_fn = FinetuneDinov2CollateFn(self.args, self.image_processor)
        return DataLoader(
            self.test_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn
        )

    def configure_optimizers(self):
        if self.args['task'] == 'pretrain':
            optimiser = torch.optim.AdamW(self.parameters(), lr=self.args['pt_lr'])
            lr_scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.1, patience=5)
            return {
                "optimizer": optimiser,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': 'val_loss',
                    'frequency': 1 
                }
            }
        else:
            pretrain_main_params, finetune_main_params = [], []
            if self.args['load'] is not None:
                checkpoint = torch.load(self.args['load'])['state_dict']
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    if name in checkpoint:
                        pretrain_main_params.append(param)
                    else:
                        finetune_main_params.append(param)
            else:
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    finetune_main_params.append(param)

            optimiser = torch.optim.AdamW(
                [{'params': pretrain_main_params, 'lr': self.args['pt_lr']},
                 {'params': finetune_main_params, 'lr': self.args['ft_lr']}])

            lr_scheduler = ReduceLROnPlateau(optimiser, mode='max', factor=0.1, patience=5)
            return {
                "optimizer": optimiser,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': self.args['monitor_metric'],
                    'frequency': 1 
                }
            }

    def tokenization(self, text, pair_text=None, device=None):
        if pair_text is None:
            inputs = self.tokenizer(text, padding=True, return_tensors='pt', return_token_type_ids=True,
                                    max_length=self.args['max_length'] + 1,
                                    truncation=True)
        else:
            inputs = self.tokenizer(text, pair_text, padding=True, return_token_type_ids=True,
                                    return_tensors='pt', max_length=200, truncation=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        return inputs

    def obtain_decoder_input_ids(self, inputs):
        decoder_input_ids = inputs['input_ids']
        decoder_attention_mask = inputs['attention_mask'][:, :-1] 
        label_ids = decoder_input_ids[:, 1:].detach().clone()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_input_ids[decoder_input_ids == self.tokenizer.sep_token_id] = self.tokenizer.pad_token_id
        return decoder_input_ids, decoder_attention_mask, label_ids

    def obtain_reference_reports(self, text):
        inputs = self.tokenizer(text, padding=True, max_length=self.args['max_length'],
                                truncation=True, return_tensors='pt')
        ref_reports = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        return ref_reports


    def multiview_fusion_network(self, image_embed, patient_ids, batch_size, view_positions):
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(int)
        labels = torch.from_numpy(labels)
        labels.fill_diagonal_(0)

        new_image_embed = []
        for i in range(batch_size):
            if labels[i].sum() == 0:
                new_image_embed.append(image_embed[i])
                continue
            multiview_image_embed = torch.cat([image_embed[j] for j, tag in enumerate(labels[i]) if tag == 1], dim=0)
            cur_image_embed = self.fusion_multiview(image_embed[i], multiview_image_embed,
                                                    multiview_image_embed)

            new_image_embed.append(cur_image_embed)
        new_image_embed = torch.stack(new_image_embed, dim=0)
        return new_image_embed

    def text_encoder_forward(self, inputs):
        text_embed = self.text_encoder(**inputs)
        text_embed = self.text_projection(text_embed['last_hidden_state']) 
        return text_embed

    def image_encoder_forward(self, images, view_positions):
        outputs = self.image_encoder(images)
        image_embed = torch.cat([outputs['pooler_output'].unsqueeze(dim=1), outputs['last_hidden_state']], dim=1)
        valid_view_positions = [vp.split('_')[0] for vp in view_positions]
        image_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in valid_view_positions]
        image_embed = torch.cat(image_pos_embed, dim=0) + image_embed
        image_embed = self.ln_1(image_embed)
        image_embed = self.image_projection(image_embed)  

        return image_embed

    def forward(self, images, patient_ids, view_positions, indications, prior_reports, reports=None, mode='train'):
        device = images.device
        batch_size = len(indications)

        prompt_embed = None
        if self.args['is_indication']:
            if self.args['is_prior_report']:
                prompt_inputs = self.tokenization(indications, pair_text=prior_reports, device=device)
                prompt_embed = self.text_encoder_forward(prompt_inputs)
            else:
                prompt_inputs = self.tokenization(indications, pair_text=None, device=device)
                prompt_embed = self.text_encoder_forward(prompt_inputs)
        else:
            if self.args['is_prior_report']:
                prompt_inputs = self.tokenization(prior_reports, pair_text=None, device=device)
                prompt_embed = self.text_encoder_forward(prompt_inputs)

        image_embed = self.image_encoder_forward(images, view_positions)
        ori_image_embed = image_embed[:batch_size] + torch.cat([self.type_pos_embed[0].unsqueeze(0)] * batch_size,
                                                               dim=0)

        temporal_pos_embed = []
        for vp in view_positions:
            if 'prior' not in vp:
                temporal_pos_embed.append(self.temp_pos_embed[0].unsqueeze(0))
            else:
                if 'latest' in vp:
                    temporal_pos_embed.append(self.temp_pos_embed[1].unsqueeze(0))
                else:  
                    temporal_pos_embed.append(self.temp_pos_embed[2].unsqueeze(0))
        image_embed = image_embed + torch.cat(temporal_pos_embed, dim=0)
        image_embed = self.ln_2(image_embed)
        if self.args['is_multiview_learning']:

            image_embed = self.multiview_fusion_network(image_embed, patient_ids, batch_size, view_positions)
        else:
            image_embed = image_embed[:batch_size]

        image_embed = image_embed + torch.cat([self.type_pos_embed[1].unsqueeze(0)] * batch_size, dim=0)
        image_embed = torch.cat([ori_image_embed, image_embed], dim=1)
        image_embed = self.ln_3(image_embed)

        if prompt_embed is not None:
            encoder_attention_mask = torch.ones(image_embed.size()[:2], dtype=torch.long).to(device)
            extended_image_masks = get_extended_attention_mask(encoder_attention_mask, encoder_attention_mask.size())
            extended_text_masks = get_extended_attention_mask(prompt_inputs['attention_mask'], prompt_embed.size())

            x, y = image_embed.clone(), prompt_embed
            for layer_idx, image_layer in enumerate(self.fusion_multimodal):
                x1 = image_layer(x, y, attention_mask=extended_image_masks,
                                 encoder_attention_mask=extended_text_masks, output_attentions=True)
                x = x1[0]
            encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=x)
        else:
            encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=image_embed)
        if mode == 'train':
            report_inputs = self.tokenization(reports, device=device)
            decoder_input_ids, decoder_attention_mask, labels_ids = self.obtain_decoder_input_ids(report_inputs)

            outputs = self.text_decoder.encoder_decoder(
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                return_dict=True,
                labels=labels_ids
            )
            return outputs['loss']
        else:
            outputs = self.generate(encoder_outputs)
            generated_reports = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return generated_reports

    def generate(self, encoder_outputs):

        outputs = self.text_decoder.encoder_decoder.generate(

            max_length=self.args['max_length'],
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=self.args['num_beams'],
            return_dict_in_generate=True,
            use_cache=True,
            encoder_outputs=encoder_outputs,
        )

        return outputs['sequences']

    def training_step(self, batch, batch_idx):
        image_ids, images, reports, patient_ids, view_positions, indications, prior_reports = batch
        loss = self(images, patient_ids, view_positions, indications, prior_reports, reports=reports, mode='train')

        self.log_dict({'lm_loss': loss}, on_step=True, on_epoch=True, batch_size=len(reports),
                      prog_bar=True, sync_dist=True)
        self.train_loss_metric.update(loss)
        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            self.mylog.info(
                f"Epoch {self.current_epoch}, training step {batch_idx}/{self.trainer.num_training_batches}, "
                f"{loss.detach().item()}, lr: {self.optimizers().param_groups[0]['lr']},"
                f"{self.optimizers().param_groups[1]['lr']}")
        return loss

    def validation_step(self, batch, batch_idx):
        image_ids, images, reports, patient_ids, view_positions, indications, prior_reports = batch
        generated_reports = self(images, patient_ids, view_positions, indications, prior_reports,
                                 reports=None, mode='sample')
        generated_reports = [text if len(text) > 0 else "..." for text in generated_reports]
        reference_reports = self.obtain_reference_reports(reports)  

        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            self.mylog.info(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}")

        self.val_report_logger.update(generated_reports, dicom_ids=image_ids, labels=reference_reports)

        self.val_f1chexbert_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.val_coco_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.val_radgraph_metrics.update(generated_reports, reference_reports, ids=image_ids)

    def test_step(self, batch, batch_idx):

        image_ids, images, reports, patient_ids, view_positions, indications, prior_reports = batch
        start = time.time()
        generated_reports = self(images, patient_ids, view_positions, indications, prior_reports,
                                 reports=None, mode='sample')
        end = time.time()
        self.time_sum += end - start
        reference_reports = self.obtain_reference_reports(reports)

        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            self.mylog.info(
                f"Testing step {batch_idx}/{self.trainer.num_test_batches[0]}")

        self.test_report_logger.update(generated_reports, dicom_ids=image_ids, labels=reference_reports)

        self.test_f1chexbert_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.test_coco_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.test_radgraph_metrics.update(generated_reports, reference_reports, ids=image_ids)

    def on_train_epoch_end(self):
        epoch_loss = self.train_loss_metric.compute()
        self.train_loss_metric.reset()
        self.mylog.info(
            f"Epoch {self.current_epoch}, Training is over, "
            f"epoch lm_loss = {epoch_loss}, lr: {self.optimizers().param_groups[0]['lr']}, "
            f"{self.optimizers().param_groups[1]['lr']}"
            "\n###############################################################"
        )

    def on_validation_epoch_end(self):
        self.val_report_logger.compute(self.current_epoch)
        self.val_report_logger.reset()
        scores = {}
        output = self.val_radgraph_metrics.compute()
        scores.update(output)
        self.val_radgraph_metrics.reset()

        output = self.val_f1chexbert_metrics.compute()
        scores.update(output)
        self.val_f1chexbert_metrics.reset()

        output = self.val_coco_metrics.compute()
        scores.update(output)
        self.val_coco_metrics.reset()

        scores['RB'] = scores['F1-Radgraph-partial'] + scores['bleu_4']
        scores['RC'] = scores['F1-Radgraph-partial'] + scores['chexbert_all_micro_f1']
        scores['RCB'] = scores['F1-Radgraph-partial'] + scores['bleu_4'] + scores['chexbert_all_micro_f1']

        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)

        if scores[self.args['monitor_metric']] > self.val_best_scores['best_monitor_metric']:
            self.val_best_scores = {
                "best_epoch": self.current_epoch,
                'best_monitor_metric': scores[self.args['monitor_metric']]
            }

        metrics_item = '\n'.join([f'{k}: {v}' for k, v in scores.items()])
        self.mylog.info(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, Validation is over, current metrics:\n"
            f"best validation epoch: {self.val_best_scores['best_epoch']}, "
            f"best val_metrics: {self.args['monitor_metric']} = {self.val_best_scores['best_monitor_metric']}\n"
            f"{metrics_item} \n"
        )

    def on_test_epoch_end(self):

        print(f"all time is {self.time_sum}, the average time of each image is {self.time_sum / len(self.test_set)}")

        self.test_report_logger.log(1)
        self.test_report_logger.compute(self.current_epoch)
        self.test_report_logger.reset()

        scores = {}
        output = self.test_radgraph_metrics.compute()
        scores.update(output)
        self.test_radgraph_metrics.reset()

        output = self.test_f1chexbert_metrics.compute()
        scores.update(output)
        self.test_f1chexbert_metrics.reset()

        output = self.test_coco_metrics.compute()
        scores.update(output)
        self.test_coco_metrics.reset()

        scores['RB'] = scores['F1-Radgraph-partial'] + scores['bleu_4']
        scores['RC'] = scores['F1-Radgraph-partial'] + scores['chexbert_all_micro_f1']
        scores['RCB'] = scores['F1-Radgraph-partial'] + scores['bleu_4'] + scores['chexbert_all_micro_f1']

        print('\n')
        print(scores)

        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)
        metrics_item = '\n'.join([f'{k}: {v}' for k, v in scores.items()])
        self.mylog.info(
            "###############################################################\n"
            f"test is over, current metrics:"
            f"{metrics_item} \n"
        )

def sinkhorn_transport(cost, eps=0.05, n_iter=50):
    B, K = cost.shape
    cost_row_min = cost.min(dim=1, keepdim=True)[0]
    K_mat = torch.exp(-(cost - cost_row_min) / eps)
    a = torch.ones(B, device=cost.device) / B
    b = torch.ones(K, device=cost.device) / K
    u = torch.ones(B, device=cost.device)
    v = torch.ones(K, device=cost.device)
    K_t = K_mat.t()
    for _ in range(n_iter):
        u = a / (K_mat @ v + 1e-12)
        v = b / (K_t @ u + 1e-12)
    P = (u.unsqueeze(1) * K_mat) * v.unsqueeze(0)
    P_rownorm = P * B
    P_rownorm = P_rownorm / (P_rownorm.sum(dim=1, keepdim=True) + 1e-12)
    return P_rownorm