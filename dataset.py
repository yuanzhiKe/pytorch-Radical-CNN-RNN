import os, fileinput
from torch.utils.data import Dataset


def RakutenReviewRaw(review_dir):
    """
    Return a iterator to load raw Rakuten Review
    """
    file_path_list = [os.path.join(review_dir, filename) for filename in os.listdir(review_dir)]
    return fileinput.input(file_path_list)