#!/usr/bin/python3

from os.path import isfile, join
import json
import torch

from config import RecommendTrainConfig, GlobalConfig
from config import DatasetConfig
from dataset.dataset import Dataset
from lib.recommend_train import recommend_train
from utils import load_pkl
from collections import namedtuple
from transformers import BertTokenizer, BertModel
from config.constants import *
import torch.nn as nn
import argparse
if RecommendTrainConfig.using_learn:
    from widget.dst_aware_model_learn import DST_AWARE
else:
    from widget.dst_aware_model import DST_AWARE
from torchvision.models import resnet18

CommonData = namedtuple('CommonData',
                        ['image_paths'])

def train(args):
    """Train model.

    Args:
        task (int): Task.
        model_file_name (str): Model file name (saved or to be saved).

    """

    # tokenizer & add special_tokens
    with open(DatasetConfig.special_tokens, "rb") as handle:
        special_tokens_dict = json.load(handle)

    tokenizer = BertTokenizer.from_pretrained(PRETRAIN_TEXT_ENCODER)
    tokenizer.add_special_tokens(special_tokens_dict)

    # state keys
    state_keys = tokenizer(MMD_SLOT_TOKENS, return_tensors='pt', add_special_tokens=DatasetConfig.add_special_tokens)
    state_keys = state_keys['input_ids'].squeeze()

    # Dialog data files.
    if RecommendTrainConfig.full_data:
        # common data
        # Check if data exists.
        if not isfile(DatasetConfig.wodst_common_raw_data_file):
            raise ValueError('No common raw data.')
        # Load extracted common data.
        common_data = load_pkl(DatasetConfig.wodst_common_raw_data_file)

        train_dialog_data_file = DatasetConfig.wodst_recommend_train_dialog_file
        valid_dialog_data_file = DatasetConfig.wodst_recommend_valid_dialog_file
        test_dialog_data_file = DatasetConfig.wodst_recommend_test_dialog_file
    else:
        # common data
        # Check if data exists.
        if not isfile(DatasetConfig.common_raw_data_file):
            raise ValueError('No common raw data.')
        # Load extracted common data.
        common_data = load_pkl(DatasetConfig.common_raw_data_file)

        # dialogue data
        train_dialog_data_file = DatasetConfig.recommend_train_dialog_file
        valid_dialog_data_file = DatasetConfig.recommend_valid_dialog_file
        test_dialog_data_file = DatasetConfig.recommend_test_dialog_file

    if not isfile(train_dialog_data_file):
        raise ValueError('No train dialog data file.')
    if not isfile(valid_dialog_data_file):
        raise ValueError('No valid dialog data file.')
    if not isfile(test_dialog_data_file):
        raise ValueError('No test dialog data file.')

    # Load extracted dialogs.
    train_dialogs = load_pkl(train_dialog_data_file)
    valid_dialogs = load_pkl(valid_dialog_data_file)
    test_dialogs = load_pkl(test_dialog_data_file)

    # Dataset wrap.
    train_dataset = Dataset(
        'train',
        common_data.image_paths,
        train_dialogs,
        tokenizer)
    valid_dataset = Dataset(
        'valid',
        common_data.image_paths,
        valid_dialogs,
        tokenizer)
    test_dataset = Dataset(
        'test',
        common_data.image_paths,
        test_dialogs,
        tokenizer)

    print('Train dataset size:', len(train_dataset))
    print('Valid dataset size:', len(valid_dataset))
    print('Test dataset size:', len(test_dataset))

    # dst train dataset
    if RecommendTrainConfig.few_shot:
        fewshot_dialog_data_file = DatasetConfig.recommend_fewshot_dialog_file
        if not isfile(fewshot_dialog_data_file):
            raise ValueError('No dew shot dialog data file.')
        fewshot_dialogs = load_pkl(fewshot_dialog_data_file)
        fewshot_dataset = Dataset(
            'fewshot',
            common_data.image_paths,
            fewshot_dialogs,
            tokenizer)
        print('few shot dataset size:', len(fewshot_dataset))
    else:
        fewshot_dataset = None

    # pretained embedding
    pretrained_embedding = BertModel.from_pretrained(PRETRAIN_TEXT_ENCODER)
    pretrained_embedding.resize_token_embeddings(len(tokenizer))
    pretrained_embedding.vocab_size = len(tokenizer)
    pretrained_embedding = pretrained_embedding.embeddings

    # pretrained image encoder
    pretrained_image_encoder = resnet18(pretrained = True)
    pretrained_image_encoder = nn.Sequential(
        *list(pretrained_image_encoder.children())[:-1]
    )

    # init model.
    encoder = DST_AWARE(pretrained_embedding, pretrained_image_encoder, state_keys)
    model_file = join(DatasetConfig.dump_dir, args['model_file_name'])

    recommend_train(
        encoder,
        train_dataset,
        valid_dataset,
        test_dataset,
        model_file,
        fewshot_dataset
    )

def parse_cmd():
    """Parse commandline parameters.

    Returns:
        Dict[str, List[str]]: Parse result.

    """

    # Definition of argument parser.
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('model_file_name', metavar='<model_file_name>')
    parser.add_argument('device', metavar='<device>')

    # Namespace -> Dict
    parse_res = vars(parser.parse_args())
    return parse_res

def main():
    # Parse commandline parameters and standardize.
    parse_result = parse_cmd()
    GlobalConfig.device = torch.device(parse_result['device'] if torch.cuda.is_available() else "cpu")
    train(parse_result)

if __name__ == '__main__':
    main()
