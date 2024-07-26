# pretain test encoder
PRETRAIN_TEXT_ENCODER = 'bert-base-uncased'
# Speakers
USER_SPEAKER = 0
SYS_SPEAKER = 1
PADDING_ID = 0
# model hyper parameters
TEXT_EMB_SIZE = 768
CROSS_ATTN_HEAD_NUM = 8
CROSS_LAYER_NUM = 3
CO_CROSS_LAYER_NUM = 3
HOPS = 3

ATTRIBUTES_MAX_LEN = 110

MMD_SLOT_TOKENS = ['age', 'category', 'brand', 'size', 'length', 'color',
                   'fit', 'style', 'care', 'type', 'material', 'gender', 'print',
                   'like', 'dislike']

MMD_DIALOGUE_STATE = {
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

SIMMC_SLOT_TOKENS = ['review', 'brand', 'length', 'size', 'pattern', 'price',
                     'color', 'material', 'rate', 'type']

SIMMC_DIALOGUE_STATE = {
    'customerReview': 'none',
    'brand': 'none',
    'sleeveLength': 'none',
    'availableSizes': 'none',
    'pattern': 'none',
    'price': 'none',
    'color': 'none',
    # 'size': 'none',
    'materials': 'none',
    'customerRating': 'none',
    'type': 'none'
}