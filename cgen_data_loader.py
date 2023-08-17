import os
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm, trange
import torch
from torch.utils.data import Dataset

class DataLoader(Dataset):
    def __init__(self, data, images_path, name="dataset"):
        self.images = []
        self.captions = []

        print("\nLoading images and labels for the " + name + "...")
        image_names = list(data.keys())
        for image_name in tqdm(image_names):
            # Image normalization
            path = os.path.join(images_path, image_name)
            image = Image.open(path)
            image = image.convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = preprocess(image)
            self.images.append(image)
            self.captions.append(data[image_name])

    def __getitem__(self, i):
        return self.images[i], self.captions[i]

    def __len__(self):
        return len(self.images)


class DataLoader2(Dataset):
    def __init__(self, data, images_vectors, name="dataset"):
        self.images = []
        self.captions = []

        print("\nLoading images and labels for the " + name + "...")
        image_names = list(data.keys())
        for image_name in tqdm(image_names):
            self.images.append(torch.FloatTensor(images_vectors[image_name]).squeeze(0))
            self.captions.append(data[image_name])

    def __getitem__(self, i):
        return self.images[i], self.captions[i]

    def __len__(self):
        return len(self.images)


def create_targets(caption, word2vec):
    # Define the expected continuation given the first token's word embedding
    try:
        token_target_ids = word2vec[caption[1]].unsqueeze(0)
    except:
        token_target_ids = word2vec['<unk>'].unsqueeze(0)

    # Define the expected continuation given the proceeding token's word embedding
    for i in range(2, len(caption)):
        try:
            token_target_ids = torch.cat((word2vec[caption[i]].unsqueeze(0), token_target_ids))
        except:
            token_target_ids = torch.cat((word2vec['<unk>'].unsqueeze(0), token_target_ids))

    # Define the expected continuation (end of sequence) given the last token's word embedding
    token_target_ids = torch.cat((word2vec['endofsequence'].unsqueeze(0), token_target_ids))

    # Return the target embeddings
    return token_target_ids


class KnnDataLoader:

    def __init__(self, data, images_path, name="dataset"):
        self.images_path = images_path
        self.image_names = []
        self.captions = []

        print("\nLoading images and labels for the " + name + "...")
        image_names = list(data.keys())
        for image_name in tqdm(image_names):
            self.image_names = image_name
            self.captions.append(data[image_name])

    def get(self, image_name):
        # Image normalization
        path = os.path.join(self.images_path, image_name)
        image = Image.open(path)
        image = image.convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image)

        return image

    def len(self):
        return len(self.images)
