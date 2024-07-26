from utils.better_abc import ABCMeta, abstract_attribute


class ValidConfig(metaclass=ABCMeta):
    batch_size = abstract_attribute()
    num_batches = abstract_attribute()
    num_data_loader_workers = abstract_attribute()

class RecommendValidConfig(ValidConfig):
    batch_size = 64 # 8
    num_batches = 128
    num_data_loader_workers = 2 # 1

