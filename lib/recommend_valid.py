import torch
from torch.utils.data import DataLoader
import sys
import os

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from config import RecommendValidConfig, RecommendTrainConfig
from dataset.dataset import Dataset
from lib.metric import cal_recall, cal_precision, cal_NDCG
from utils import collate

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def recommend_valid(
        encoder,
        valid_dataset):

    """Recommend valid.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Gat.
        similarity (Similarity): Intention.
        valid_dataset (Dataset): Valid dataset.

    """

    # Valid dataset loader.
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=RecommendValidConfig.batch_size,
        shuffle=True,
        num_workers=RecommendValidConfig.num_data_loader_workers,
        collate_fn = collate
    )

    # Switch to eval mode.
    encoder.eval()
    RecommendTrainConfig.DST = False

    sum_loss = 0
    num_batches = 0

    sum_NDCG_list = [0 for _ in range(2)]
    sum_Precision_list = [0 for _ in range(2)]
    sum_Recall_list = [0 for _ in range(2)]
    k = [1, 5]

    with torch.no_grad():
        for batch_id, valid_data in enumerate(tqdm(valid_data_loader, ncols=80)):
            # Only valid `ValidConfig.num_batches` batches.
            if batch_id >= RecommendValidConfig.num_batches:
                break

            context_dialog, pos_products, neg_products = valid_data

            # Encode context.
            loss, rank_temp, num_pos_products = encoder(context_dialog, pos_products, neg_products, eval = True)

            sum_loss += loss

            num_batches += 1
            # print("count : {}".format(num_batches))

            for j in k:
                sum_Recall_list[int(j/5)] += cal_recall(j, rank_temp , num_pos_products , len(num_pos_products))
                sum_Precision_list[int(j/5)] += cal_precision(j, rank_temp , num_pos_products , len(num_pos_products))
                sum_NDCG_list[int(j/5)] += cal_NDCG(j, rank_temp, num_pos_products, len(num_pos_products))
        for j in k:
            print("\ntotal avg Recall@{} = {}".format(j, sum_Recall_list[int(j / 5)] / num_batches))
            print("\ntotal avg Precision@{} = {}".format(j, sum_Precision_list[int(j / 5)] / num_batches))
            print("\ntotal avg NDCG@{} = {}".format(j, sum_NDCG_list[int(j / 5)] / num_batches))

    # Switch to train mode.
    encoder.train()
    if RecommendTrainConfig.few_shot==False:
        RecommendTrainConfig.DST = True

    return sum_loss / num_batches, sum_NDCG_list[-1]
