# sklearn and nltk imports
from sklearn.model_selection import KFold
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import nltk
import os
import pickle

from cnn_backbone import label_to_one_hot

"""
George Z., Phoivos C., Panagiotis C. code
"""

nltk.download("punkt", quiet=True)

# tensorflow imports
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer

# progress bar
from tqdm import tqdm
import math

# utils imports
from utils.text_handler import TextHandler
from utils.vocabulary import Vocabulary
import time


class Dataset:
    def __init__(self, image_vectors: dict, captions_data: dict, clear_long_captions: bool = True):
        """ Base class to create the employed dataset for my research, i.e. ImageCLEF and IU X-Ray

        Args:
            image_vectors (dict): Dictionary with keys to be the ImageIDs and values the image embeddings.
            captions_data (dict): Dictionary with keys to be the ImageIDs and values the captions.
            clear_long_captions (bool, optional): If we want to drop the outlier long captions. Defaults to True.
        """
        self.image_vectors = image_vectors
        self.captions_data = captions_data
        self.clear_long_captions = clear_long_captions
        # init a text handler object to pre-process training captions
        self.text_handler = TextHandler()

    def delete_long_captions(self, data: dict, threshold: int = 80) -> dict:
        """ Function that removes the long captions only from the training set. This method was utilised during ImageCLEF campaign.

        Args:
            data (dict): Dictionary with keys to be the ImageIDs and values the captions.
            threshold (int, optional): The maximum length limit. Defaults to 80.

        Returns:
            dict: Dictionary with keys to be the ImageIDs and values the captions, without the instances whose captions are long.
        """
        filtered_data = {}

        for image_id, caption in data.items():
            tokens = word_tokenize(caption)
            if len(tokens) <= threshold:
                filtered_data[image_id] = caption

        return filtered_data

    # @staticmethod
    def build_splits(self) -> tuple[list, list, list]:
        """ This function makes the split sets for training, validation and test.
        In particulare, we followed the next splits:
        train: 80%
        valid: 5%
        test: 15%

        Returns:
            tuple[list, list, list]: Training, validation, test set ids.
        """

        image_ids = list(self.captions_data.keys())

        print('len image ids:', len(image_ids))
        # print(image_ids)
        # np.random.shuffle(image_ids)

        train_path = '/media/georg_mosh/Samsung_T51/ImageCLEFmedical/pkaliosis_splits/imageclef2023_captions_train_mysplit.csv'
        val_path = '/media/georg_mosh/Samsung_T51/ImageCLEFmedical/pkaliosis_splits/imageclef2023_captions_valid_mysplit.csv'
        test_path = '/media/georg_mosh/Samsung_T51/ImageCLEFmedical/pkaliosis_splits/imageclef2023_captions_test_mysplit.csv'
        hidden_test_path = '/media/georg_mosh/Samsung_T51/ImageCLEFmedical/pkaliosis_splits/imageclef2023_captions_hidden_test.csv'

        clef_df_train = pd.read_csv(train_path, sep=',', encoding='latin')
        clef_df_val = pd.read_csv(val_path, sep=',', encoding='latin')
        clef_df_test = pd.read_csv(test_path, sep=',', encoding='latin')
        clef_df_hidden_test = pd.read_csv(hidden_test_path, sep=',', encoding='latin')

        train = list(clef_df_train.ID)
        dev = list(clef_df_val.ID)
        test = list(clef_df_test.ID)
        hidden_test = list(clef_df_hidden_test.ID)

        # train = train + dev[:len(dev)-1000]
        # dev = dev[len(dev)-1000:]

        # train= train + dev[:len(dev)-1000]
        # dev = dev[len(dev)-1000:]
        # test = test
        # hidden_test = hidden_test

        # train = train + test[:9500]   # todo: ask Panagiotis is it useful?
        # test = test[9500:]

        """split_point1, split_point2 = math.floor(0.75 * len(image_ids)), math.floor(0.85 * len(image_ids))
        train, dev, test = (
            image_ids[:split_point1],
            image_ids[split_point1:split_point2],
            image_ids[split_point2:],
        )

        dev_split_threshold = int(0.1 * len(train))
        train, dev = (
            train[:-dev_split_threshold],
            train[-dev_split_threshold:],s
        )"""

        """for i, item in enumerate(train):
            train[i] = item + '.jpg'

        for i, item in enumerate(dev):
            dev[i] = item + '.jpg'

        for i, item in enumerate(test):
            test[i] = item + '.jpg'"""

        print("in dataset.py len train:", len(train))
        print("in dataset.py len val:", len(dev))
        print("in dataset.py len test:", len(test))
        return train, dev, test, hidden_test

    # @staticmethod
    def get_image_vectors(self, keys: list) -> dict:
        """ Fetches from the whole dataset the image embeddings according to the utilised set.

        Args:
            keys (list): Split set ids

        Returns:
            dict: Dictionary with keys to be the ImageIDs and values the image embeddings, for each split set.
        """

        for i in range(len(keys)):
            keys[i] = keys[i] + '.jpg'

        return {k: v for k, v in tqdm(self.image_vectors.items(), desc="Fetching image embeddings..") if k in keys}

    def get_captions(self, _ids: list) -> dict:
        for i in range(len(_ids)):
            _ids[i] = _ids[i].replace(".jpg", "")
        return {key: value for key, value in self.captions_data.items() if key in _ids}

    def build_pseudo_cv_splits(self) -> tuple[list, list]:
        """ This function makes cross-validaion splis using K-Fold cross validation. It was used only for ImageCLEF campaign.
        More details are described in my Thesis.

        Returns:
            tuple[list, list]: Training and test fold sets.
        """
        image_ids = list(self.captions_data.keys())
        np.random.shuffle(image_ids)

        # apply 15-Fold CV
        kf = KFold(n_splits=15)
        train_fold_ids, test_fold_ids = list(), list()
        for train_index, test_index in kf.split(image_ids):
            train_ids = [image_ids[index] for index in train_index]
            test_ids = [image_ids[index] for index in test_index]
            train_fold_ids.append(train_ids)
            test_fold_ids.append(test_ids)

        return train_fold_ids, test_fold_ids

    def build_vocab(self, training_captions: list, threshold: int = 3) -> tuple[Vocabulary, Tokenizer, dict, dict]:
        """ This method creates the employed vocabulary given the training captions

        Args:
            training_captions (list): All training captions
            threshold (int, optional): The cut-off frequence for Vocabulary. Defaults to 3.

        Returns:
            tuple[Vocabulary, Tokenizer, dict, dict]: The Vocabulary object, the fitted tokenizer, the word-to-idx dictionary, and idx-to-word dictionary.
            The latters are mappers for words and index respectively
        """
        vocab = Vocabulary(texts=training_captions, threshold=threshold)
        tokenizer, word2idx, idx2word = vocab.build_vocab()
        return vocab, tokenizer, word2idx, idx2word


class IuXrayDataset(Dataset):
    def __init__(self, image_vectors: dict, captions_data: dict, tags_data: dict):
        """ Child class to create the employed IU X-Ray, inheriting the base class methods

        Args:
            image_vectors (dict): Dictionary with keys to be the ImageIDs and values the image embeddings.
            captions_data (dict): Dictionary with keys to be the ImageIDs and values the captions.
            tags_data (dict): Dictionary with keys to be the ImageIDs and values the tags embeddings.
        """
        super().__init__(image_vectors=image_vectors, captions_data=captions_data, clear_long_captions=False)
        self.tags_data = tags_data
        # get the splits
        self.train_dataset, self.dev_dataset, self.test_dataset = self.build_dataset()
        # build linguistic attributes
        self.vocab, self.tokenizer, self.word2idx, self.idx2word = super().build_vocab(
            training_captions=list(self.train_dataset[1].values()))

    def __str__(self) -> str:
        """ Python built-in function for prints

        Returns:
            str: A modified print.
        """
        text = f"Train: patients={len(self.train_dataset[0])}, captions={len(self.train_dataset[1])}, tags={len(self.train_dataset[2])}"
        text += f"\nDev: patients={len(self.dev_dataset[0])}, captions={len(self.dev_dataset[1])}, tags={len(self.dev_dataset[2])}"
        text += f"\nTest: patients={len(self.test_dataset[0])}, captions={len(self.test_dataset[1])}, tags={len(self.test_dataset[2])}"
        return text

    def get_splits_sets(self) -> tuple[list, list, list]:
        """ Fetches the data for each split set.

        Returns:
            tuple[list, list, list]: train_dataset, dev_dataset, test_dataset
        """
        return self.train_dataset, self.dev_dataset, self.test_dataset

    def get_tokenizer_utils(self) -> tuple[Vocabulary, Tokenizer, dict, dict]:
        """ Fetches the linguistic utilities.

        Returns:
            tuple[Vocabulary, Tokenizer, dict, dict]:  The Vocabulary object, the fitted tokenizer, the word-to-idx dictionary, and idx-to-word dictionary.
            The latters are mappers for words and index respectively
        """
        return self.vocab, self.tokenizer, self.word2idx, self.idx2word

    def __get_tags(self, _ids: list) -> dict:
        """ Fetches from the whole dataset the tags embeddings according to the utilised set.

        Args:
            _ids (list): Split set ids

        Returns:
            dict: Dictionary with keys to be the ImageIDs and values the tags embeddings
        """

        return {key: value for key, value in self.tags_data.items() if key in _ids}

    def build_dataset(self) -> tuple[list, list, list]:
        """ Begins the whole process for the dataset creation.

        Returns:
            tuple[list, list, list]: The training dataset, the validation dataset and the test dataset for our models.
            All sets are in list format.
            1st index --> image vectors
            2nd index --> captions
            3rd index --> tags
        """
        # random split
        train_ids, dev_ids, test_ids = super().build_splits()

        # fetch images
        train_images = super().get_image_vectors(train_ids)
        dev_images = super().get_image_vectors(dev_ids)
        test_images = super().get_image_vectors(test_ids)
        # fetch captions
        train_captions = super().get_captions(train_ids)
        dev_captions = super().get_captions(dev_ids)
        test_captions = super().get_captions(test_ids)
        # apply preprocess to training captions
        train_captions_prepro = self.text_handler.preprocess_all(list(train_captions.values()))

        train_captions_prepro = dict(zip(train_ids, train_captions_prepro))
        # fetch tags
        train_tags = self.__get_tags(train_ids)
        dev_tags = self.__get_tags(dev_ids)
        test_tags = self.__get_tags(test_ids)
        # build data for each set
        train_dataset = [train_images, train_captions_prepro, train_tags]
        dev_dataset = [dev_images, dev_captions, dev_tags]
        test_dataset = [test_images, test_captions, test_tags]

        return train_dataset, dev_dataset, test_dataset


class ImageCLEFDataset(Dataset):
    def __init__(self, image_vectors: dict, captions_data: dict):
        """_summary_

        Args:
            image_vectors (dict): _description_
            captions_data (dict): _description_
        """
        super().__init__(image_vectors=image_vectors, captions_data=captions_data, clear_long_captions=True)
        self.train_dataset, self.dev_dataset, self.test_dataset, self.hidden_test = self.build_dataset()

        print("in dataset.py len train2:", len(self.train_dataset[0]), type(self.train_dataset[0]))
        print("in dataset.py len val2:", len(self.dev_dataset[0]), type(self.dev_dataset[0]))
        print("in dataset.py len test2:", len(self.test_dataset[0]), type(self.test_dataset[0]))

        time.sleep(3)

        self.vocab, self.tokenizer, self.word2idx, self.idx2word = super().build_vocab(
            training_captions=list(self.train_dataset[1].values()))

    def __str__(self) -> str:
        """ Python built-in function for prints

        Returns:
            str: A modified print.
        """
        text = f"Train: patients={len(self.train_dataset[0])}, captions={len(self.train_dataset[1])}"
        text += f"\nDev: patients={len(self.dev_dataset[0])}, captions={len(self.dev_dataset[1])}"
        text += f"\nTest: patients={len(self.test_dataset[0])}, captions={len(self.test_dataset[1])}"
        return text

    def get_splits_sets(self) -> tuple[list, list, list]:
        """ Fetches the data for each split set.

        Returns:
            tuple[list, list, list]: train_dataset, dev_dataset, test_dataset
        """
        return self.train_dataset, self.dev_dataset, self.test_dataset, self.hidden_test

    def get_tokenizer_utils(self) -> tuple[Vocabulary, Tokenizer, dict, dict]:
        """ Fetches the linguistic utilities.

        Returns:
            tuple[Vocabulary, Tokenizer, dict, dict]:  The Vocabulary object, the fitted tokenizer, the word-to-idx dictionary, and idx-to-word dictionary.
            The latters are mappers for words and index respectively
        """
        return self.vocab, self.tokenizer, self.word2idx, self.idx2word

    def build_dataset(self) -> tuple[list, list, list]:
        """ Begins the whole process for the dataset creation.

        Returns:
            tuple[list, list, list]: The training dataset, the validation dataset and the test dataset for our models.
            All sets are in list format.
            1st index --> image vectors
            2nd index --> captions
        """
        # random split
        train_ids, dev_ids, test_ids, hidden_test_ids = super().build_splits()  # <-- small changes here...

        print('in build_dataset train ids:', len(train_ids))
        print('in build_dataset val ids:', len(dev_ids))
        print('in build_dataset test ids:', len(test_ids))
        print('in build_dataset hidden test ids:', len(hidden_test_ids))

        time.sleep(5)

        # fetch images
        train_images = super().get_image_vectors(train_ids)  # same...
        dev_images = super().get_image_vectors(dev_ids)  # same...
        test_images = super().get_image_vectors(test_ids)  # same...
        hidden_test_images = super().get_image_vectors(hidden_test_ids)  # same...
        # fetch captions
        train_captions = super().get_captions(train_ids)
        dev_captions = super().get_captions(dev_ids)
        test_captions = super().get_captions(test_ids)
        hidden_test_captions = super().get_captions(hidden_test_ids)

        # remove long outlier captions from training set
        """train_modified_captions = super().delete_long_captions(data=train_captions)
        # get new training ids after removing
        train_new_ids = list(train_modified_captions.keys())"""
        """for k in range(len(train_ids)):
            train_ids[k] = train_new_ids[k] + '.jpg'"""

        """train_new_images = {
            key:image_vector for key, image_vector in train_images.items() if key in train_new_ids
        }"""
        # apply preprocess to training captions
        train_captions_prepro = self.text_handler.preprocess_all(
            list(train_captions.values()))

        train_captions_prepro = dict(zip(train_ids, train_captions_prepro))
        # build data for each set
        train_dataset = [train_images, train_captions_prepro]
        # train_dataset = [train_images, train_captions]
        dev_dataset = [dev_images, dev_captions]

        test_dataset = [test_images, test_captions]
        hidden_test_dataset = [hidden_test_images, hidden_test_captions]

        return train_dataset, dev_dataset, test_dataset, hidden_test_dataset



def load_encoded_vecs(filename):
    """ Loads the image embeddings for each image id, we extracted offline during my research

    Args:
        filename (str): the whole path of npy file

    Returns:
        dict: encoded_vectors from filename
    """
    with open(filename, 'rb') as f:
        print("Image Encoded Vectors loaded from directory path:", filename)
        #print("PICKLES:", pickle.load(f))
        return pickle.load(f)


def load_imageclef_data():
    """ Loads ImageCLEF dataset from directory

    Returns:
        tuple[dict, dict]: Image vectors, captions in dictionary format, with keys to be the Image IDs.
    """
    # get dataset path
    imageclef_data_path = os.path.join("/media/georg_mosh/Samsung_T51/ImageCLEFmedical/pkaliosis_splits")
    print('imclefdatapath:', imageclef_data_path)
    # fetch images, captions
    imageclef_image_captions_pairs = os.path.join(imageclef_data_path, 'Imageclef2023_dataset_all.csv')
    clef_df = pd.read_csv(imageclef_image_captions_pairs, sep=',', encoding='latin')
    # clef_df = clef_df.drop(columns = ['Unnamed: 0'])
    print('column names:', clef_df.columns)
    captions = dict(zip(clef_df.ID.to_list(), clef_df.caption.to_list()))

    imageclef_image_captions_pairs_hidden = os.path.join(imageclef_data_path,
                                                         'imageclef2023_captions_hidden_test.csv')
    hidden_test_clef_df = pd.read_csv(imageclef_image_captions_pairs_hidden, sep=',', encoding='latin')
    captions_hidden_test = dict(zip(hidden_test_clef_df.ID.to_list(), hidden_test_clef_df.caption.to_list()))

    captions.update(captions_hidden_test)

    print('Length captions:', len(captions))

    encoder = "efficientnet0-v2"

    image_encoded_vectors_path = os.path.join(imageclef_data_path, f"{encoder}.pkl")
    print("IMAGE ENCODED VECTORS PATH:", image_encoded_vectors_path)
    # load image embeddings for the employed encoder
    image_vecs = load_encoded_vecs(image_encoded_vectors_path)
    return image_vecs, captions


def create_imageCLEF_dataset(images:dict, captions:dict) -> ImageCLEFDataset:
    """ Builds the ImageCLEF dataset using the ImageCLEFDataset loader class

    Args:
        images (dict): Dictionary with keys to be the ImageIDs and values the image embeddings.
        captions (dict): Dictionary with keys to be the ImageIDs and values the captions.

    Returns:
        ImageCLEFDataset: the employed ImageCLEFDataset object
    """
    imageCLEF_dataset = ImageCLEFDataset(image_vectors=images, captions_data=captions)
    return imageCLEF_dataset


def create_dataset():
    """
    Interface to colleagues code.
    -------
    ImageCLEFDataset: the employed ImageCLEFDataset object
    """
    # case ImageCLEF
    image_vecs, captions = load_imageclef_data()
    imageCLEF_dataset = create_imageCLEF_dataset(image_vecs, captions)
    imageCLEF_dataset.word2vec = label_to_one_hot(imageCLEF_dataset.word2idx)
    return imageCLEF_dataset
