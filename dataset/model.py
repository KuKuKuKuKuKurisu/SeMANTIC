"""Data models.

Data models:
    * Utterance
    * Product
    * TidyUtterance

"""
import torch

from config.dataset_config import DatasetConfig
from utils import pad_or_clip_images

class Utterance():
    """Utterance data model.

    Attributes:
        speaker (int): Speaker.
                       0 (USER_SPEAKER) for user, 1 (SYS_SPEAKER) for system.
        utter_type (int): Utterance type.
        text (str): Text.
        pos_images (List[int]): Positive images.
        neg_images (List[int]): Negative images.

    """

    def __init__(self, speaker, utter_type, text,
                 pos_images, neg_images, dst):
        self.speaker = speaker
        self.utter_type = utter_type
        self.text = text
        self.pos_images = pos_images
        self.neg_images = neg_images
        self.dst = dst

    def __repr__(self):
        return str((self.speaker, self.utter_type, self.text,
                    self.pos_images, self.neg_images, self.dst))


class Product():
    """Product data model.

    Attributes:
        product_name (str): Product name, which is the name of the .json file.
        attribute_dict (Dict[str, Any]): Attribute dictionary.

    """

    def __init__(self, product_name, attribute_dict):
        self.product_name = product_name
        self.attribute_dict = attribute_dict

class TidyUtterance():
    """Tidy utterance data model.

    Attributes:
        utter_type (int): Utterance type.
        text (List[int]): Text.
        text_len (int): Text length.
        pos_images (List[int]): Positive images.
        pos_images_num (int): Number of positive images.
        neg_images (List[int]): Negative images.
        neg_images_num (int): Number of negative images.

    """

    def __init__(self, utter):
        self.speaker = utter.speaker
        self.utter_type = utter.utter_type
        self.text = utter.text
        self.pos_images, self.pos_images_num = pad_or_clip_images(
            utter.pos_images, DatasetConfig.max_pos_num)
        self.neg_images, self.neg_images_num = pad_or_clip_images(
            utter.neg_images, DatasetConfig.max_neg_num)
        if self.pos_images_num!=0:
            self.pos_images_weights = torch.tensor([2.] * self.pos_images_num + [0.] * (DatasetConfig.max_pos_num - self.pos_images_num))
        else:
            self.pos_images_weights = torch.tensor([0.] * (DatasetConfig.max_pos_num - self.pos_images_num))
        if self.neg_images_num!=0:
            self.neg_images_weights = torch.tensor([2.] * self.neg_images_num + [0.] * (DatasetConfig.max_neg_num - self.neg_images_num))
        else:
            self.neg_images_weights = torch.tensor([0.] * (DatasetConfig.max_neg_num - self.neg_images_num))
        self.dst = utter.dst

    def __repr__(self):
        return str((self.utter_type,
                    self.text,
                    self.pos_images, self.pos_images_num,
                    self.neg_images, self.neg_images_num,
                    self.pos_images_weights, self.neg_images_weights,
                    self.dst))