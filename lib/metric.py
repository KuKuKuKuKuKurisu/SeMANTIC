import torch
import math
from config.global_config import GlobalConfig
from config.dataset_config import DatasetConfig

def cal_recall(k, rank_temp, num_pos_products, batch_size):

    '''Recall'''
    recall_list = []
    for j in range(batch_size):
        num = 0
        try:
            for i in range(num_pos_products[j].item()):
                if(rank_temp[i][j] < k):
                    num += 1
            recall_list.append(num / num_pos_products[j].item())
        except:
            print(1)
    # average
    sum_recall = 0
    for i in range(batch_size):
        sum_recall += recall_list[i]
    avg_recall = sum_recall / batch_size
    return avg_recall

def cal_precision(k, rank_temp, num_pos_products, batch_size):
    '''Precision '''
    # Precision
    bili_list = []
    for j in range(batch_size):
        num = 0
        for i in range(num_pos_products[j].item()):
            if(rank_temp[i][j] < k):
                num += 1
        bili_list.append(num / k)
    # average
    bili_sum = 0
    for i in range(batch_size):
        bili_sum += bili_list[i]
    avg_pre = bili_sum / batch_size
    return avg_pre

def cal_NDCG(k, rank_temp, num_pos_products, batch_size):
    '''NDCG'''
    # relevance
    dcg_matrix = torch.zeros(DatasetConfig.max_pos_num + DatasetConfig.max_neg_num,
                             batch_size, dtype=torch.long).to(GlobalConfig.device)
    for j in range(batch_size):
        for i in range(num_pos_products[j].item()):
            dcg_matrix[rank_temp[i][j]][j] += 1
    # DCG
    dcg_list = []
    for j in range(batch_size):
        dcg_sum = 0
        for i in range(k):
            result = dcg_matrix[i][j].item() / math.log2(i+2)
            dcg_sum += result
        dcg_list.append(dcg_sum)
    # IDCG
    sorted_dcg_list = []
    for j in range(batch_size):
        sorted_dcg_sum = 0
        sorted_list = dcg_matrix.cpu().numpy()
        sorted_list = sorted(sorted_list[:,j].tolist(),reverse=True)
        for i in range(k):
            b_result = sorted_list[i] / math.log2(i+2)
            sorted_dcg_sum += b_result
        sorted_dcg_list.append(sorted_dcg_sum)
    # NDCG
    NDCG_list = []
    for i in range(batch_size):
        NDCG_list.append(dcg_list[i]/sorted_dcg_list[i])
    # average NDCG
    sum_NDCG = 0
    for i in range(batch_size):
        sum_NDCG += NDCG_list[i]
    avg_NDCG = sum_NDCG / batch_size

    return avg_NDCG