"""Dataset module."""
import json
from os.path import join, isfile
import re
import torch
from PIL import Image
from torch.utils import data
from config import RecommendTrainConfig
if RecommendTrainConfig.simmc2:
    from config.simmc_dataset_config import SimmcDatasetConfig as DatasetConfig
else:
    from config import DatasetConfig
from utils import get_product_path
from tqdm import tqdm

class Dataset(data.Dataset):
    """Dataset class."""

    # Constants.
    EMPTY_IMAGE = torch.zeros(3, DatasetConfig.image_size,
                              DatasetConfig.image_size)
    EMPTY_PRODUCT_TEXT = ''

    def __init__(self, task, image_paths, dialogs, tokenizer):
        self.task = task
        self.image_paths = image_paths
        self.dialogs = dialogs
        self.tokenizer = tokenizer
        if RecommendTrainConfig.simmc2:
            with open(DatasetConfig.product_data_directory, "rb") as f:
                self.item_attrs = json.load(f)

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index: int):
        """ Get item for a given index.

        Args:
            index (int): item index.

        Returns:

            - RECOMMEND_TASK
                context_dialog:
                    texts: Texts (dialog_context_size + 1, dialog_text_max_len).
                    text_lengths: Text lengths (dialog_context_size + 1, ).
                    images: Images (dialog_context_size + 1, pos_images_max_num,
                                    3, image_size, image_size).
                    utter_type (int): The type of the last user utterance.
                pos_products:
                    num_pos_products (int): Number of positive products.
                    pos_images: Positive images
                                (pos_images_max_num, 3, image_size, image_size).
                    pos_product_texts: Positive product texts
                                     (pos_images_max_num, product_text_max_len).
                    pos_product_text_lengths: Positive product text lengths
                                          (pos_images_max_num, ).
                neg_products:
                    num_neg_products (int): Number of negative products.
                    neg_images: Negative images
                                (neg_images_max_num, 3, image_size, image_size).
                    neg_product_texts: Negative product texts
                                     (neg_images_max_num, product_text_max_len).
                    neg_product_text_lengths: Negative product text lengths
                                          (neg_images_max_num, ).

        """
        dialog = self.dialogs[index % len(self.dialogs)]

        context_dialog = self._get_context_dialog(dialog[:DatasetConfig.dialog_context_size])
        # Products (Text & Image).
        utter = dialog[-1]  # System response.
        pos_products, pos_sample_id, pos_valid_images_paths = self._get_images_product_texts(utter.pos_images, utter.pos_images_num, utter.pos_images_weights, DatasetConfig.max_pos_num)
        neg_products, neg_sample_id, neg_valid_images_paths = self._get_images_product_texts(utter.neg_images, utter.neg_images_num, utter.neg_images_weights, DatasetConfig.max_neg_num)

        return context_dialog, pos_products, neg_products, pos_valid_images_paths, neg_valid_images_paths

    def _get_context_dialog(self, dialog):
        """Get context dialog.

        Note: The last utterance of the context dialog is system response.

        Args:
            dialog (TidyDialog): Dialog.

        Returns:
            texts: Texts (dialog_context_size, dialog_text_max_len).
            text_lengths: Text lengths (dialog_context_size, ).
            images: Images (dialog_context_size, pos_images_max_num, 3,
                           image_size, image_size).
            utter_type (int): The type of the last user utterance.

        """
        # Text.
        text_ids = []
        text_masks = []
        text_type = []
        turn_mask = []

        for utter_id, utter in enumerate(dialog[:-1]):
            text_id = utter.text['input_ids']
            text_mask = utter.text['attention_mask']
            text_speaker = utter.speaker
            if text_speaker == 'user':
                text_speaker = [self.tokenizer.added_tokens_encoder['[@user]']] * DatasetConfig.dialog_text_max_len
            elif text_speaker == 'system':
                text_speaker = [self.tokenizer.added_tokens_encoder['[@system]']] * DatasetConfig.dialog_text_max_len

            text_ids.append(text_id)
            text_masks.append(text_mask)
            text_type.append(text_speaker)
            if utter.utter_type=='pad':
                # pad_id = 0
                turn_mask.append(0)
            else:
                turn_mask.append(1)

        # query
        query = dialog[-1]
        text_ids.append(query.text['input_ids'])
        text_masks.append(query.text['attention_mask'])
        text_type.append([self.tokenizer.added_tokens_encoder['[@query]']] * DatasetConfig.dialog_text_max_len)
        turn_mask.append(1)

        texts = {
            "input_ids": text_ids,
            "attention_mask": text_masks,
            "text_type": text_type,
            "turn_mask": turn_mask
        }

        # Image.
        image_list = [[] for _ in range(DatasetConfig.dialog_context_size)]
        image_mask = [[] for _ in range(DatasetConfig.dialog_context_size)]

        for idx, utter in enumerate(dialog):
            for img_id in utter.pos_images:

                if RecommendTrainConfig.simmc2 == False:
                    path = self.image_paths[img_id]
                else:
                    path = self.image_paths[img_id] + '.jpg'

                if path:
                    path = join(DatasetConfig.image_data_directory, path)
                else:
                    path = ''

                if path and isfile(path):
                    try:
                        raw_image = Image.open(path).convert("RGB")
                        image = DatasetConfig.transform(raw_image)
                        image_list[idx].append(image)
                        image_mask[idx].append(1)
                    except OSError:
                        image_list[idx].append(Dataset.EMPTY_IMAGE)
                        image_mask[idx].append(0)
                else:
                    image_list[idx].append(Dataset.EMPTY_IMAGE)
                    image_mask[idx].append(0)

        images = torch.stack(list(map(torch.stack, image_list)))
        # (dialog_context_size, pos_images_max_num, 3, image_size, image_size)

        # Utterance type.
        utter_type = dialog[-2].utter_type

        # dst
        dst = dialog[-1].dst

        # dst['text_type'] = [self.tokenizer.added_tokens_encoder['[@state]']] * DatasetConfig.dst_max_len

        return texts, images, image_mask, utter_type, dst

    def _get_images_product_texts(self, image_ids, image_num, image_weight, max_image_num):
        """Get images and product texts of a response.

        Args:
            image_ids (List[int]): Image ids.
            num_products (int): Number of real images.

        Returns:
            num_products (int): Number of products (exclude padding).
            images: Images (num_products, 3, image_size, image_size).
            product_texts: Product texts (num_products, product_text_max_len).
            product_text_lengths: Product text lengths (num_products, ).

        """

        images = []
        product_texts = []
        img_sample_id = torch.tensor([])

        valid_images_paths = []
        if self.task != 'test':
            if image_num >= max_image_num:
                img_sample_id = torch.multinomial(image_weight, max_image_num)
                img_samples = torch.tensor(image_ids)[img_sample_id].tolist()
            else:
                img_samples = image_ids[:max_image_num]
        else:
            img_samples = image_ids[:max_image_num]

        for img_id in img_samples:
            if img_id == 0:
                images.append(self.EMPTY_IMAGE)
                product_texts.append(self.EMPTY_PRODUCT_TEXT)
                valid_images_paths.append('')
                continue

            if RecommendTrainConfig.simmc2 == False:
                image_name = self.image_paths[img_id]
                product_path = get_product_path(image_name)
            else:
                image_name = self.image_paths[img_id] + '.jpg'
                product_path = self.image_paths[img_id]

            image_path = join(DatasetConfig.image_data_directory, image_name)
            valid_images_paths.append(image_path)
            # Image.
            raw_image = Image.open(image_path).convert("RGB")
            image = DatasetConfig.transform(raw_image)

            images.append(image)

            # Text.
            text = self._get_product_text(product_path)
            product_texts.append(text)

        # To tensors.
        images = torch.stack(images)
        product_texts = self.tokenizer(product_texts, padding='max_length', max_length = DatasetConfig.product_text_max_len, truncation=True, return_tensors='pt',
                                       add_special_tokens=DatasetConfig.add_special_tokens)
        del product_texts['token_type_ids']
        product_texts['text_type'] = torch.tensor([[self.tokenizer.added_tokens_encoder['[@item_attrs]']] * DatasetConfig.product_text_max_len] * len(images))

        if image_num >= max_image_num:
            valid_image_num = max_image_num
        else:
            valid_image_num = image_num

        return (valid_image_num, images, product_texts), img_sample_id, valid_images_paths

    def _get_product_text(self, product_path):
        if RecommendTrainConfig.simmc2:
            product_texts = self.item_attrs[product_path].strip().lower()
            product_texts = re.sub(r"[ \( \) \[ \] \< \> , = : ;]", " ", product_texts)
        else:
            ignore_key_list = ['details', 'available_sizes', 'bestSellerRank', 'review',
                               'care', 'avgStars', 'reviewStars']

            product_dict = json.load(open(product_path))
            texts = []
            for key, value in product_dict.items():
                # Note: Only a space is also empty.
                if key in ignore_key_list:
                    continue
                if value is not None and value != '' and value != ' ':
                    if value.lower() != 'unk':
                        texts.extend([key, value])

            product_texts = ' '.join(texts).replace('/', ' ').lstrip('‘').rstrip('’').strip().lower()
            product_texts = re.sub(r"[ \( \) \[ \] \< \> , = : ;]", " ", product_texts)

        return product_texts