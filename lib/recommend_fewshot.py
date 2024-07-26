import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from dataset.dataset import Dataset
from config.train_config import RecommendTrainConfig
from utils import collate
from tqdm import tqdm

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def recommend_fewshot(
        encoder,
        fewshot_dataset,
        optimizer,
        epochs
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

    # Data loader.
    fewshot_data_loader = DataLoader(
        dataset=fewshot_dataset,
        batch_size=RecommendTrainConfig.batch_size,
        shuffle=True,
        num_workers=RecommendTrainConfig.num_data_loader_workers,
        collate_fn = collate
    )
    RecommendTrainConfig.DST = True
    RecommendTrainConfig.start_learn = True
    for epoch in range(epochs):
        print("few-shot epoch:{}".format(epoch+1))
        for batch_id, train_data in enumerate(tqdm(fewshot_data_loader)):
            # Sets gradients to 0.
            optimizer.zero_grad()

            context_dialog, pos_products, neg_products = train_data

            # Encode context.
            loss, loss_splits = encoder(context_dialog, pos_products, neg_products)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), RecommendTrainConfig.max_gradient_norm)
            optimizer.step()

    RecommendTrainConfig.DST = False
    RecommendTrainConfig.start_learn = False