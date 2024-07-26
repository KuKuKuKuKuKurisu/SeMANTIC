from utils.better_abc import ABCMeta, abstract_attribute

class TrainConfig(metaclass=ABCMeta):
    batch_size = abstract_attribute()
    num_iterations = abstract_attribute()
    learning_rate = abstract_attribute()
    print_freq = abstract_attribute()
    valid_freq = abstract_attribute()
    patience = abstract_attribute()
    num_data_loader_workers = abstract_attribute()

class RecommendTrainConfig(TrainConfig):

    # training on simmc data or not
    simmc2 = False
    # using semi-supervised learning or not
    using_learn = True

    # few shot args
    few_shot = False
    fewshot_epochs = 1
    # when training on MMD-v3, low dst_data_proportion indicates less ds labeled MMD-v3 data is used
    dst_data_proportion = 0.01
    # when training on MMD-v2, low wodst_data_proportion indicates less ds unlabeled MMD-v2 data is used
    wodst_data_proportion = 0.01

    # training on MMD-v2 or not
    full_data = False

    batch_size = 64
    num_iterations = 50

    if simmc2:
        learning_rate = 4e-5 #85
        print_freq = 64
        valid_freq = 128
    else:
        learning_rate = 5e-4
        print_freq = 128
        valid_freq = 256

    patience = 50
    fine_tune_patience = 5
    num_data_loader_workers = 4
    gradient_clip = False
    max_gradient_norm = 1
    image_encoder = 'resnet18'

    # support losses
    # text image sim loss
    txt_img_loss = True
    txt_img_threshold = 1.0
    text_image_sim_loss_weight = 1

    # kl loss
    kl_diff_loss = True
    diff_loss_weight = 1

    # learn loss
    start_learn = False
    start_learn_epoch = 1
    learn_loss_weight = 0.5

    # fix args
    DST = True
    dropout = 0.1
    soft_max_temperture = 0.5