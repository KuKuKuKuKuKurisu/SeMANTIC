import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import isfile
from config import RecommendTestConfig, GlobalConfig, RecommendTrainConfig
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils import collate
from dataset.dataset import Dataset
from lib.metric import cal_recall, cal_precision, cal_NDCG
import json
import shutil

def recommend_test(
        encoder,
        test_dataset,
        model_file):

    """Recommend test.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Gat.
        similarity (Similarity): Intention.
        test_dataset (Dataset): test dataset.

    """

    # Test dataset loader.
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=RecommendTestConfig.batch_size,
        num_workers=RecommendTestConfig.num_data_loader_workers,
        collate_fn = collate
    )

    # Load saved state.
    if isfile(model_file):
        print('loading best model...')
        state = torch.load(model_file)
        encoder.load_state_dict(state['context_text_encoder'])
    encoder.to(GlobalConfig.device)

    # Switch to eval mode.
    encoder.eval()
    RecommendTrainConfig.DST = False
    # There might be a bug in the implement of resnet.

    sum_NDCG_list = [0 for _ in range(5)]
    sum_Precision_list = [0 for _ in range(5)]
    sum_Recall_list = [0 for _ in range(5)]
    count = 0
    k = [1, 5, 10, 15, 20]
    # k = [5, 10, 20]
    batch_count = len(test_dataset) / RecommendTestConfig.batch_size
    print("batch count :{}".format(batch_count))

    # error analysis
    EA = {}

    with torch.no_grad():
        for batch_id, test_data in enumerate(tqdm(test_data_loader, ncols=80)):
            # Only valid `ValidConfig.num_batches` batches.

            context_dialog, pos_products, neg_products, pos_valid_images_paths, neg_valid_images_paths = test_data

            loss, rank_temp, num_pos_products, pred_pos_sims, pred_neg_sims = encoder(context_dialog, pos_products, neg_products, eval = True)

            # all_valid_paths = pos_valid_images_paths[0]
            # all_valid_paths.extend(neg_valid_images_paths[0])
            #
            # all_pred_sims = pred_pos_sims
            # all_pred_sims.extend(pred_neg_sims)
            # item_sim_path_pairs = []
            # for i in range(len(all_valid_paths)):
            #     if all_valid_paths[i]!='':
            #         item_sim_path_pairs.append([all_valid_paths[i], all_pred_sims[i].item()])
            #
            # item_sim_path_pairs = sorted(item_sim_path_pairs, key=lambda x:x[1], reverse=True)
            # exp_output_dir = 'exp_output/{}'.format(batch_id)
            # if os.path.exists(exp_output_dir)==False:
            #     os.mkdir(exp_output_dir)
            # for item in item_sim_path_pairs:
            #     shutil.copy(item[0], exp_output_dir)
            #     if len(os.listdir(exp_output_dir))==10:
            #         break

            count += 1
            print("count : {}".format(count))

            recall_record = []
            presicion_record = []
            NDCG_record = []

            # if RecommendTestConfig.batch_size==1:
            #     ndcg5 = cal_NDCG(5, rank_temp , num_pos_products , len(num_pos_products))
            #     EA[batch_id] = {
            #         'ndcg@5':ndcg5,
            #         'rank':rank_temp.squeeze().tolist()
            #     }

            for j in k:
                sum_Recall_list[int(j/5)] += cal_recall(j, rank_temp , num_pos_products , len(num_pos_products))
                sum_Precision_list[int(j/5)] += cal_precision(j, rank_temp , num_pos_products , len(num_pos_products))
                sum_NDCG_list[int(j/5)] += cal_NDCG(j, rank_temp, num_pos_products, len(num_pos_products))
                print("total avg Recall@{} = {}".format(j, sum_Recall_list[int(j/5)] / count))
                print("total avg Precision@{} = {}".format(j, sum_Precision_list[int(j/5)] / count))
                print("total avg NDCG@{} = {}".format(j, sum_NDCG_list[int(j/5)] / count))

                recall_record.append("total avg Recall@{} = {}".format(j, sum_Recall_list[int(j/5)] / count))
                presicion_record.append("total avg Precision@{} = {}".format(j, sum_Precision_list[int(j/5)] / count))
                NDCG_record.append("total avg NDCG@{} = {}".format(j, sum_NDCG_list[int(j/5)] / count))

            with open(model_file[:-3]+'txt','a') as f:
                f.write('batch: {}: recall: {} precision: {} NDCG: {}\n'.format(batch_id,
                                                                                str(recall_record), str(presicion_record), str(NDCG_record)))

    if len(EA)>0:
        with open(model_file[:-4]+'_EA.json','w') as f:
            json.dump(EA, f, ensure_ascii=False, indent=1)
