import torch
import torch.nn as nn
import numpy as np

from models.models import classifier
from models.loss import SupConLoss, ConditionalEntropyLoss, NTXentLoss


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class CoTMix(Algorithm):
    def __init__(self, backbone_fe, configs, hparams, device):
        super(CoTMix, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)


        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        
        self.contrastive_loss = NTXentLoss(device, hparams["batch_size"], 0.2, True)
        self.entropy_loss = ConditionalEntropyLoss()
        self.sup_contrastive_loss = SupConLoss(device)

    def update(self, src_x, src_y, trg_x):

# ====== Temporal Mixup =====================
        mix_ratio = round(self.hparams.mix_ratio, 2)
        temporal_shift = self.hparams.temporal_shift
        h = temporal_shift // 2 # half


        src_dominant = mix_ratio * src_x + (1 - mix_ratio) * \
                     torch.mean(torch.stack([torch.roll(trg_x, -i, 2) for i in range(-h, h)], 2), 2)

        trg_dominant = mix_ratio * trg_x + (1 - mix_ratio) * \
                     torch.mean(torch.stack([torch.roll(src_x, -i, 2) for i in range(-h, h)], 2), 2)


# ====== Extract features and calc logits =====================
        self.optimizer.zero_grad()

        # Src original features
        src_orig_feat = self.feature_extractor(src_x)
        src_orig_logits = self.classifier(src_orig_feat)
        
        # Target original features
        trg_orig_feat = self.feature_extractor(trg_x)
        trg_orig_logits = self.classifier(trg_orig_feat)


# -----------  The two main losses: L_CE on source and L_ent on target 
        # Cross-Entropy loss
        src_cls_loss = self.cross_entropy(src_orig_logits, src_y)
        loss = src_cls_loss * round(self.hparams.src_cls_weight, 2)

        # Target Entropy loss
        trg_entropy_loss = self.entropy_loss(trg_orig_logits)
        loss += trg_entropy_loss * round(self.hparams.trg_entropy_weight, 2)


# -----------  Auxiliary losses
        # Extract source-dominant mixup features.
        src_dominant_feat = self.feature_extractor(src_dominant)
        src_dominant_logits = self.classifier(src_dominant_feat)

        # supervised contrastive loss on source domain side
        src_concat = torch.cat([src_orig_logits.unsqueeze(1), src_dominant_logits.unsqueeze(1)], dim=1)
        src_supcon_loss = self.sup_contrastive_loss(src_concat, src_y)
        #src_con_loss = self.contrastive_loss(src_orig_logits, src_dominant_logits) # unsupervised_contrasting
        loss += src_supcon_loss * round(self.hparams.src_supCon_weight, 2)



        # Extract target-dominant mixup features.
        trg_dominant_feat = self.feature_extractor(trg_dominant)
        trg_dominant_logits = self.classifier(trg_dominant_feat)

        # Unsupervised contrastive loss on target domain side
        trg_con_loss = self.contrastive_loss(trg_orig_logits, trg_dominant_logits)
        loss += trg_con_loss * round(self.hparams.trg_cont_weight, 2)


        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(),
                'src_cls_loss': src_cls_loss.item(),
                'trg_entropy_loss': trg_entropy_loss.item(),
                'src_supcon_loss': src_supcon_loss.item(),
                'trg_con_loss': trg_con_loss.item()
                }
