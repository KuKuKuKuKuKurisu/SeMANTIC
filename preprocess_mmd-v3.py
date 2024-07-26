# -*- coding: utf-8 -*-
import json
import os
from os.path import isfile
import re
from tqdm import tqdm
from utils import save_pkl, load_pkl, process_txt
from config import RecommendTrainConfig, constants
from config import DatasetConfig
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
    # Get URL to filename mapping dict.
    with open(DatasetConfig.url2img, 'r') as file:
        url_image_pairs = [line.strip().split(' ') for line in file.readlines()]
    url_image_pairs = [(p[0], p[1]) for p in url_image_pairs]
    url2img = dict(url_image_pairs)

    # Divided it into two steps.
    # URL -> Path => URL -> index & index -> Pathtrain
    # Element of index 0 should be empty image.
    image_url_id = {'': 0}
    image_paths = ['']
    for url, img in url2img.items():
        image_url_id[url] = len(image_url_id)
        image_paths.append(img)

    return image_url_id, image_paths

def preprocess_mmd_dst(dst):
    result = {
        'age': 'none',
        'product_category': 'none',
        'brand': 'none',
        'sizes': 'none',
        'length': 'none',
        'colors': 'none',
        'fit': 'none',
        'styles': 'none',
        'care': 'none',
        'types': 'none',
        'materials': 'none',
        'gender': 'none',
        'print': 'none',
        'likes': 'none',
        'dislikes': 'none'
    }

    inform_data = dst.get('informable', [])
    request_data = dst.get('requestable', [])

    for inform_d in inform_data:
        slots = inform_d.get('slot', [])
        values = inform_d.get('values', [])
        sentiments = inform_d.get('sentiment', [])

        if len(slots) == 0:
            slots = ['none']
        if len(values) == 0:
            values = ['none']
        if len(sentiments) == 0:
            sentiments = ['none']

        slots = ' '.join(slots)
        values = ' '.join(values)
        sentiments = ' '.join(sentiments)

        slots = slots.split('>>')

        s = process_txt(slots[0])

        se = process_txt(sentiments)

        v = process_txt(values)

        if s in result.keys():
            if result[s] == 'none':
                result[s] = se + ' ' + v
            else:
                result[s] = result[s] + ' ' + se + ' ' + v

    likes = []
    dislikes = []
    for request_d in request_data:
        likes.extend(request_d.get('likes', []))
        dislikes.extend(request_d.get('dislikes', []))

    if len(likes) == 0:
        likes = ['none']
    if len(dislikes) == 0:
        dislikes = ['none']

    likes = ' '.join(likes)
    dislikes = ' '.join(dislikes)

    result['likes'] = process_txt(likes)
    result['dislikes'] = process_txt(dislikes)

    return result

def main():

    #common data
    splits = ['train', 'valid', 'test']

    with open(DatasetConfig.image_id_file) as fp:
        url_id = json.load(fp)[0]
    id_url = {x:y for y,x in url_id.items()}

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
                    dst = utterance.get('ds', {})
                    dst = preprocess_mmd_dst(dst)

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
                    new_pos_images = [id_url[x] for x in images]
                    new_neg_images = [id_url[x] for x in false_images]

                    pos_images = []
                    for img in new_pos_images:
                        try:
                            pos_images.append(image_url_id[img])
                        except:
                            pass

                    neg_images = []
                    for img in new_neg_images:
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