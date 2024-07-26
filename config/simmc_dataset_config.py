"""Dataset configurations."""
import config

"""Dataset configurations."""
import pwd
from os.path import join, isdir, isfile
from torchvision import transforms
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)


class SimmcDatasetConfig():
    """Dataset configurations."""

    #父目录路径
    data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../simmc2"))
    # data_directory = join(data_directory, 'dataset')

    # dialog_directory = join(data_directory, 'MMD2withDST_dst_cleaned_only')
    dialog_directory = data_directory

    # dataset path
    train_dialog_data_directory = join(dialog_directory, 'train/')
    valid_dialog_data_directory = join(dialog_directory, 'valid/')
    test_dialog_data_directory = join(dialog_directory, 'test/')

    # MMD material path
    # MMD_material_path = join(data_directory, '../MMD/dataset')
    SIMMC_material_path = data_directory

    # image path
    image_data_directory = join(SIMMC_material_path, 'images')

    # image attrs path
    product_data_directory = join(SIMMC_material_path, 'img_attrs/img_attrs.json')

    dump_dir = join(data_directory, '../DST_AWARE/simmc_data')

    common_raw_data_file = join(dump_dir, 'common_raw_data.pkl')

    train_raw_data_file = join(dump_dir, 'train_raw_data.pkl')
    valid_raw_data_file = join(dump_dir, 'valid_raw_data.pkl')
    test_raw_data_file = join(dump_dir, 'test_raw_data.pkl')
    fewshot_raw_data_file = join(dump_dir, 'fewshot_raw_data.pkl')

    recommend_train_dialog_file = join(dump_dir,
                                       'recommend_train_dialog_file.pkl')
    recommend_valid_dialog_file = join(dump_dir,
                                       'recommend_valid_dialog_file.pkl')
    recommend_test_dialog_file = join(dump_dir,
                                      'recommend_test_dialog_file.pkl')
    recommend_fewshot_dialog_file = join(dump_dir,
                                         'recommend_fewshot_dialog_file.pkl')

    special_tokens = 'special_tokens.json'

    tensorboard_file = 'tensorboard/'

    dialog_context_size = 5

    dialog_text_max_len = 30
    product_text_max_len = 30
    dst_max_len = 30
    state_value_max_len = 6
    add_special_tokens = False

    clip_max_len=77

    max_pos_num = 5
    max_neg_num = 1000

    resize_image_size = 72
    image_size = 64

    transform = transforms.Compose([
        transforms.Resize([resize_image_size, resize_image_size]),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])