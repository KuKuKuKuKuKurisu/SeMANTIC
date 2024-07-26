from datetime import datetime
from os.path import isfile
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from dataset.dataset import Dataset
from lib.recommend_valid import recommend_valid
from lib.recommend_test import recommend_test
from lib.recommend_fewshot import recommend_fewshot
from tensorboardX import SummaryWriter
from config.train_config import RecommendTrainConfig
from config.global_config import GlobalConfig
if RecommendTrainConfig.simmc2:
    from config.simmc_dataset_config import SimmcDatasetConfig as DatasetConfig
else:
    from config.dataset_config import DatasetConfig
from utils import collate
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def recommend_train(
        encoder,
        train_dataset,
        valid_dataset,
        test_dataset,
        model_file,
        fewshot_dataset = None
):
    """Recommend train.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Gat.
        train_dataset (Dataset): Train dataset.
        valid_dataset (Dataset): Valid dataset.
        test_dataset (Dataset): Test dataset.
        model_file (str): Saved model file.

    """

    # DatasetConfig.max_pos_num = 5
    # DatasetConfig.max_neg_num = 1000
    # recommend_test(encoder,
    #                test_dataset,
    #                model_file)

    optimizer = Adam(encoder.parameters(), lr=RecommendTrainConfig.learning_rate)
    epoch_id = 0
    min_valid_loss = None
    batch_count = 0
    sum_loss = 0
    bad_loss_cnt = 0
    max_bad_count = 0

    max_valid_ndcg = None

    # Load saved state.
    if isfile(model_file):
        print('model exists, loading model...')
        state = torch.load(model_file)
        encoder.load_state_dict(state['context_text_encoder'])
        optimizer.load_state_dict(state['optimizer'])
        epoch_id = state['epoch_id']
        min_valid_loss = state['min_valid_loss']
        batch_count = state['batch_count']

    # Switch to train mode.
    encoder.train()
    encoder.to(GlobalConfig.device)

    # Data loader.
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=RecommendTrainConfig.batch_size,
        shuffle=True,
        num_workers=RecommendTrainConfig.num_data_loader_workers,
        collate_fn = collate
    )

    writer = SummaryWriter(model_file[:-4]+'_'+DatasetConfig.tensorboard_file)
    total_steps = len(train_data_loader) * RecommendTrainConfig.num_iterations
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps=total_steps * 0.01)

    DatasetConfig.max_pos_num = 1
    DatasetConfig.max_neg_num = 4

    if RecommendTrainConfig.few_shot:
        RecommendTrainConfig.DST = False

    for epoch_id in range(epoch_id, RecommendTrainConfig.num_iterations):

        if (RecommendTrainConfig.using_learn==True) & (RecommendTrainConfig.few_shot==False):
            if (epoch_id > 0) & (RecommendTrainConfig.start_learn == False):
                print("start learning......")
                RecommendTrainConfig.start_learn = True

        if RecommendTrainConfig.few_shot:
            recommend_fewshot(encoder, fewshot_dataset, optimizer, RecommendTrainConfig.fewshot_epochs)

        for batch_id, train_data in enumerate(tqdm(train_data_loader)):

            batch_count += 1

            # Sets gradients to 0.
            optimizer.zero_grad()

            context_dialog, pos_products, neg_products = train_data

            # Encode context.
            loss, loss_splits = encoder(context_dialog, pos_products, neg_products)

            writer.add_scalar('train loss', loss, batch_count)
            sum_loss += loss

            loss.backward()
            if RecommendTrainConfig.gradient_clip:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), RecommendTrainConfig.max_gradient_norm)
            optimizer.step()
            scheduler.step()

            # Print loss every `TrainConfig.print_freq` batches.
            if (batch_id + 1) % RecommendTrainConfig.print_freq == 0:
                sum_loss /= RecommendTrainConfig.print_freq
                print('epoch: {0}  sim loss: {1:.4f}  txt img loss: {2:.4f}  kl loss: {3:.4f} '
                      ' pred sim loss: {4:.4f}  learn loss: {5:.4f}  learn kl loss: {6:.4f}'.format(
                    epoch_id + 1, loss_splits[0], loss_splits[1], loss_splits[2],
                    loss_splits[3], loss_splits[4], loss_splits[5]))
                sum_loss = 0

            # Valid every `TrainConfig.valid_freq` batches.
            if (batch_id + 1) % RecommendTrainConfig.valid_freq == 0:

                valid_loss, valid_ndcg = recommend_valid(encoder,
                                                         valid_dataset)

                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print('valid_loss: {} \ttime: {}'.format(valid_loss, cur_time))
                writer.add_scalar('valid loss', valid_loss, batch_count)
                writer.add_scalar('ndcg@5', valid_ndcg, batch_count)

                # Save current best model.
                if min_valid_loss is None or valid_loss < min_valid_loss or max_valid_ndcg < valid_ndcg:
                    if min_valid_loss is None:
                        min_valid_loss = valid_loss
                        max_valid_ndcg = valid_ndcg

                    if valid_loss < min_valid_loss:
                        min_valid_loss = valid_loss

                    if max_valid_ndcg < valid_ndcg:
                        max_valid_ndcg = valid_ndcg

                    if max_bad_count < bad_loss_cnt:
                        max_bad_count = bad_loss_cnt

                    save_dict = {
                        'epoch_id': epoch_id,
                        'batch_count': batch_count,
                        'min_valid_loss': min_valid_loss,
                        'optimizer': optimizer.state_dict(),
                        'max_bad_count': bad_loss_cnt,

                        'context_text_encoder':
                            encoder.state_dict()
                        }
                    torch.save(save_dict, model_file)
                    print('Best model saved.')

    DatasetConfig.max_pos_num = 5
    DatasetConfig.max_neg_num = 1000
    print('testing...')
    recommend_test(encoder,
                   test_dataset,
                   model_file)
