from utils.better_abc import ABCMeta, abstract_attribute


class TestConfig(metaclass=ABCMeta):
    batch_size = abstract_attribute()
    num_data_loader_workers = abstract_attribute()

class RecommendTestConfig(TestConfig):
    batch_size = 1
    num_data_loader_workers = 2

