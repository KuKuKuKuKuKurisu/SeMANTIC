# -*- coding: utf-8 -*-
import json
import os
from os.path import isfile
import re
from tqdm import tqdm
from utils import save_pkl, load_pkl, process_txt
from config import RecommendTrainConfig
from config.simmc_dataset_config import SimmcDatasetConfig as DatasetConfig
from collections import namedtuple
from dataset.tidy_data import generate_tidy_data_file
from dataset.model import Utterance

CommonData = namedtuple('CommonData',
                        ['image_paths'])

def pre_train_word_emb(sentence):
    # remove quotation marks and spaces at begin and end
    ret = sentence.lstrip('‘').rstrip('’').strip()
    # lower characters
    ret = ret.lower()
    return ret

def get_images_path():
    """Get images (URL and filenames of local images mapping).

    URL -> Path => URL -> index & index -> Path

    Returns:
        Dict[str, int]: Image URL to index.
        List[str]: Index to the filename of the local image.

    """
    with open(DatasetConfig.product_data_directory, "rb") as f:
        url2img = json.load(f)

        # Divided it into two steps.
        # URL -> Path => URL -> index & index -> Pathtrain
        # Element of index 0 should be empty image.
    image_url_id = {'': 0}
    image_paths = ['']

    for url, img in url2img.items():
        image_url_id[url] = len(image_url_id)
        image_paths.append(url)

    return image_url_id, image_paths

def preprocess_simmc_dst(dst):
    result = {
        'customerReview': 'none',
        'brand': 'none',
        'sleeveLength': 'none',
        'availableSizes': 'none',
        'pattern': 'none',
        'price': 'none',
        'color': 'none',
        'materials': 'none',
        'customerRating': 'none',
        'type': 'none'
    }

    inform_data = dst.get('slot_values', {})

    for s, v in inform_data.items():

        v = process_txt(str(v))
        if s=='size':
            continue
        if result[s] == 'none':
            result[s] = v
        else:
            result[s] = result[s] + ' ' + v

    return result

def main():

    #common data
    splits = ['train', 'valid', 'test']

    image_url_id, image_paths = get_images_path()

    if not isfile(DatasetConfig.common_raw_data_file):
        common_data = CommonData(image_paths=image_paths)
        print('saving common_data...')
        save_pkl(common_data, 'common_data',
                 DatasetConfig.common_raw_data_file)

    for split in splits:

        if split == 'train':
            input_path = DatasetConfig.train_dialog_data_directory
            raw_output_path = DatasetConfig.train_raw_data_file
            tidy_output_path = DatasetConfig.recommend_train_dialog_file
        elif split == 'valid':
            input_path = DatasetConfig.valid_dialog_data_directory
            raw_output_path = DatasetConfig.valid_raw_data_file
            tidy_output_path = DatasetConfig.recommend_valid_dialog_file
        else:
            input_path = DatasetConfig.test_dialog_data_directory
            raw_output_path = DatasetConfig.test_raw_data_file
            tidy_output_path = DatasetConfig.recommend_test_dialog_file

        has_raw_data_pkl = isfile(raw_output_path)

        if not has_raw_data_pkl:

            dialogs = []
            for file_id, file in enumerate(tqdm(os.listdir(input_path))):
                with open(os.path.join(input_path, file), 'r') as f:
                    data = json.load(f)
                f.close()
                dialog = []
                for utterance in data:
                    # get utter attributes
                    speaker = utterance.get('speaker') #保存type
                    utter_type = f"{utterance.get('type')}"

                    # preprocess dst
                    dst = utterance.get('dst', {})
                    dst = preprocess_simmc_dst(dst)

                    utter = utterance.get('utterance')
                    text = utter.get('nlg')
                    images = utter.get('images')
                    false_images = utter.get('false images')

                    # some attributes may be empty
                    if text is None:
                        text = ""
                    if images is None:
                        images = []
                    if false_images is None:
                        false_images = []
                    if utter_type is None:
                        utter_type = ""

                    # Images
                    pos_images = []
                    for img in images:
                        try:
                            pos_images.append(image_url_id[img])
                        except:
                            pass

                    neg_images = []
                    for img in false_images:
                        try:
                            neg_images.append(image_url_id[img])
                        except:
                            pass

                    dialog.append(Utterance(speaker, utter_type, pre_train_word_emb(text), pos_images, neg_images, dst))

                dialogs.append(dialog)

            # Save common data to a .pkl file.
            save_pkl(dialogs, 'raw_{}_dialogs'.format(split), raw_output_path)
            if split=='train':
                fewshot_data_len = int(len(dialogs)*RecommendTrainConfig.dst_data_proportion)
                fewshot_dialogs = dialogs[:fewshot_data_len]
                save_pkl(fewshot_dialogs, 'raw_few_shot_dialogs', DatasetConfig.fewshot_raw_data_file)
        else:
            dialogs = load_pkl(raw_output_path)
            if split=='train':
                fewshot_dialogs = load_pkl(DatasetConfig.fewshot_raw_data_file)

        if not isfile(tidy_output_path):
            generate_tidy_data_file(dialogs, image_paths, tidy_output_path)
            if split=='train':
                generate_tidy_data_file(fewshot_dialogs, image_paths, DatasetConfig.recommend_fewshot_dialog_file)
        else:
            pass

if __name__ == '__main__':
    main()