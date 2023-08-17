import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


def cyclic(lr_min, lr_max, ns, l, t):
    delta_lr = lr_max - lr_min
    if t >= 2*l*ns and t <= (2*l+1)*ns:
        lr_current = lr_min + (t-2*l*ns)/ns * delta_lr
    elif t >= (2*l+1)*ns and t <= 2*(l+1)*ns:
        lr_current = lr_max - (t-(2*l+1)*ns)/ns * delta_lr
    if t == 2*(l+1)*ns:
        l += 1
    return lr_current, l


class DataLoader(Dataset):
    def __init__(self, class_encodings, class_indices, dataset, images_path, num_classes, name="dataset",
                 label_type="tensors", augmentation=False, random_crop=False):
        self.class_encodings = class_encodings
        self.class_indices = class_indices
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.images = []
        self.labels = []
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("\nLoading images and labels for the " + name + "...")
        if self.augmentation:
            image_names = os.listdir(images_path)
        else:
            image_names = list(dataset.keys())
        for image_name in tqdm(image_names):
            try:
                # Image normalization
                image = Image.open(os.path.join(images_path, image_name))
                image = image.convert('RGB')
                image = self.preprocess(image)
                self.images.append(image)

                if self.augmentation:
                    image_name = image_name.split("_augm_")[0]

                # Labels assignment
                if label_type == "tensors":
                    label = torch.FloatTensor(np.zeros(self.num_classes))
                    classes = dataset[image_name]
                    for cls in classes:
                        label = torch.add(label, self.class_encodings[cls])
                        # label = torch.where(label > 0, 1.0, 0.0)
                        label = torch.where(label > 0, torch.FloatTensor([1.0]), torch.FloatTensor([0.0]))
                    self.labels.append(label)
                elif label_type == "indices":
                    label = []
                    classes = dataset[image_name]
                    for cls in classes:
                        label.append(self.class_indices[cls])
                    label = torch.LongTensor(np.array(label))
                    self.labels.append(label)
                else:
                    print("Unsupported label type.")
            except:
                continue

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


class PILDataLoader(Dataset):
    def __init__(self, class_encodings, class_indices, dataset, images_path, num_classes, name="dataset",
                 label_type="tensors", augmentation=False, random_crop=False):
        self.class_encodings = class_encodings
        self.class_indices = class_indices
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.images = []
        self.labels = []

        if random_crop:
            self.preprocess = transforms.Compose([
                    transforms.RandomCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        print("\nLoading images and labels for the " + name + "...")
        if self.augmentation:
            image_names = os.listdir(images_path)
        else:
            image_names = list(dataset.keys())
        for image_name in tqdm(image_names):
            try:
                # Image normalization
                image = Image.open(os.path.join(images_path, image_name))
                image = image.convert('RGB')
                self.images.append(image)

                if self.augmentation:
                    image_name = image_name.split("_augm_")[0]

                # Labels assignment
                if label_type == "tensors":
                    label = torch.FloatTensor(np.zeros(self.num_classes))
                    classes = dataset[image_name]
                    for cls in classes:
                        label = torch.add(label, self.class_encodings[cls])
                        # label = torch.where(label > 0, 1.0, 0.0)
                        label = torch.where(label > 0, torch.FloatTensor([1.0]), torch.FloatTensor([0.0]))
                    self.labels.append(label)
                elif label_type == "indices":
                    label = []
                    classes = dataset[image_name]
                    for cls in classes:
                        label.append(self.class_indices[cls])
                    label = torch.LongTensor(np.array(label))
                    self.labels.append(label)
                else:
                    print("Unsupported label type.")
            except:
                continue

    def __getitem__(self, index):
        image = self.preprocess(self.images[index])
        return image, self.labels[index]

    def __len__(self):
        return len(self.images)


class TestDataLoader(Dataset):
    def __init__(self, class_encodings, class_indices, image_names, images_path, num_classes, name="dataset"):
        self.class_encodings = class_encodings
        self.class_indices = class_indices
        self.num_classes = num_classes
        self.images = []

        print("\nLoading images and labels for the " + name + "...")
        for image_name in tqdm(image_names):
            # Image normalization
            image = Image.open(os.path.join(images_path, image_name))
            image = image.convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = preprocess(image)
            self.images.append(image)

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)


# Parts borrowed from https://stackoverflow.com/questions/31777169/python-how-to-read-images-from-zip-file-in-memory
class DataLoaderCompressed(Dataset):
    def __init__(self, class_encodings, class_indices, dataset, images_path, num_classes, name="dataset", label_type="tensors"):
        self.class_encodings = class_encodings
        self.class_indices = class_indices
        self.num_classes = num_classes
        self.images = []
        self.labels = []

        print("\nLoading images and labels for the " + name + "...")

        image_names = list(dataset.keys())
        with zipfile.ZipFile(images_path) as images_dir:
            for image_name in tqdm(image_names):
                with images_dir.open(image_name, mode='r') as image_file:
                    image = Image.open(image_file)
                    image = image.convert('RGB')
                    preprocess = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    image = preprocess(image)
                    self.images.append(image)

                    # Labels assignment
                    if label_type == "tensors":
                        label = torch.FloatTensor(np.zeros(self.num_classes))
                        classes = dataset[image_name]
                        for cls in classes:
                            label = torch.add(label, self.class_encodings[cls])
                            label = torch.where(label > 0, 1.0, 0.0)
                        self.labels.append(label)
                    elif label_type == "indices":
                        label = []
                        classes = dataset[image_name]
                        for cls in classes:
                            label.append(self.class_indices[cls])
                        label = torch.LongTensor(np.array(label))
                        self.labels.append(label)
                    else:
                        print("Unsupported label type.")

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


class DataLoaderNoisy(Dataset):
    def __init__(self, class_encodings, class_indices, dataset, images_path, num_classes, name="dataset", label_type="tensors", augmentation=False, label_smoothing=0.0, max_jitter=0, gauss_kernel=0):
        self.class_encodings = class_encodings
        self.class_indices = class_indices
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.label_smoothing = label_smoothing
        self.gauss_kernel = gauss_kernel
        self.images_path = images_path
        self.label_type = label_type
        self.max_jitter = max_jitter
        self.dataset = dataset
        self.row_ids = {}
        self.images = []
        self.labels = []

        print("\nLoading images and labels for the " + name + "...")
        image_names = list(dataset.keys())
        for image_name in tqdm(image_names):
            # Image normalization
            image = Image.open(os.path.join(images_path, image_name))
            image = image.convert('RGB')
            self.images.append(image)

            # Labels assignment
            if label_type == "tensors":
                label = torch.FloatTensor(np.zeros(self.num_classes))
                classes = dataset[image_name]
                for cls in classes:
                    label = torch.add(label, self.class_encodings[cls])
                    label = torch.where(label > 0, torch.FloatTensor([1.0]), torch.FloatTensor([0.0]))
                    # label = torch.where(label > 0, 1.0, 0.0)
                self.labels.append(label)
            elif label_type == "indices":
                label = []
                classes = dataset[image_name]
                for cls in classes:
                    label.append(self.class_indices[cls])
                label = torch.LongTensor(np.array(label))
                self.labels.append(label)
            else:
                print("Unsupported label type.")

    def __getitem__(self, index):
        image = self.images[index]
        if self.gauss_kernel > 0 and self.max_jitter > 0:
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.ColorJitter(brightness=(0,self.max_jitter), contrast=(0,self.max_jitter), saturation=(0,self.max_jitter), hue=(0,self.max_jitter)),
                transforms.GaussianBlur(self.gauss_kernel, sigma=(0.1, 2.0)),
                transforms.RandomHorizontalFlip(p=self.noise),
                transforms.RandomVerticalFlip(p=self.noise),
            ])
        elif self.gauss_kernel > 0:
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.GaussianBlur(self.gauss_kernel, sigma=(0.1, 2.0)),
                transforms.RandomHorizontalFlip(p=self.noise),
                transforms.RandomVerticalFlip(p=self.noise),
            ])
        elif self.max_jitter > 0:
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.ColorJitter(brightness=(0,self.max_jitter), contrast=(0,self.max_jitter), saturation=(0,self.max_jitter), hue=(0,self.max_jitter)),
                transforms.RandomHorizontalFlip(p=self.noise),
                transforms.RandomVerticalFlip(p=self.noise),
            ])
        else:
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomHorizontalFlip(p=self.noise),
                transforms.RandomVerticalFlip(p=self.noise),
            ])
        image = preprocess(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.images)


class DataLoaderSlowCompressed(Dataset):
    def __init__(self, class_encodings, class_indices, dataset, images_path, num_classes, name="dataset", label_type="tensors", augmentation=False, label_smoothing=0.0):
        self.class_encodings = class_encodings
        self.class_indices = class_indices
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.label_smoothing = label_smoothing
        self.images_path = images_path
        self.label_type = label_type
        self.dataset = dataset
        self.row_ids = {}

        print("\nLoading images and labels for the " + name + "...")
        if self.augmentation:
            image_names = os.listdir(images_path)
        else:
            image_names = list(dataset.keys())
        image_index = 0
        for image_name in tqdm(image_names):
            self.row_ids[image_index] = image_name
            image_index += 1

    def __getitem__(self, index):
        # Image normalization
        image_name = self.row_ids[index]
        with zipfile.ZipFile(self.images_path) as images_dir:
            with images_dir.open(image_name, mode='r') as image_file:
                image = Image.open(image_file)
                image = image.convert('RGB')
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                image = preprocess(image)

                if self.augmentation:
                    image_name = image_name.split("_augm_")[0]

                # Labels assignment
                if self.label_type == "tensors":
                    label = torch.FloatTensor(np.zeros(self.num_classes))
                    classes = self.dataset[image_name]
                    for cls in classes:
                        label = torch.add(label, self.class_encodings[cls])
                        label = torch.where(label > 0, 1.0, 0.0)
                elif self.label_type == "indices":
                    label = []
                    classes = self.dataset[image_name]
                    for cls in classes:
                        label.append(self.class_indices[cls])
                    label = torch.LongTensor(np.array(label))
                else:
                    print("Unsupported label type.")
                
                # Label smoothing
                if self.label_smoothing > 0:
                    if self.label_type != "tensors":
                        print("Label smoothing is not supported for indices and will be discarded!")
                    else:
                        positive_classes = torch.sum(label)
                        label = torch.where(label > 0, float(1.0 - self.label_smoothing/positive_classes), 0.0)
                        complement = torch.where(label == 0, float(self.label_smoothing/(label.shape[0] - positive_classes)), 0.0)
                        label = label + complement

                return image, label

    def __len__(self):
        return len(self.row_ids)


class DataLoaderSlow(Dataset):
    def __init__(self, class_encodings, class_indices, dataset, images_path, num_classes, name="dataset", label_type="tensors", augmentation=False, label_smoothing=0.0, noise=0, max_jitter=0.5, gauss_kernel=0):
        self.class_encodings = class_encodings
        self.class_indices = class_indices
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.label_smoothing = label_smoothing
        self.gauss_kernel = gauss_kernel
        self.images_path = images_path
        self.label_type = label_type
        self.noise = noise
        self.max_jitter = max_jitter
        self.dataset = dataset
        self.row_ids = {}

        print("\nLoading images and labels for the " + name + "...")
        if self.augmentation:
            image_names = os.listdir(images_path)
        else:
            image_names = list(dataset.keys())
        image_index = 0
        for image_name in tqdm(image_names):
            self.row_ids[image_index] = image_name
            image_index += 1

    def __getitem__(self, index):
        # Image normalization
        image_name = self.row_ids[index]
        image = Image.open(os.path.join(self.images_path, image_name))
        image = image.convert('RGB')
        if self.noise == 0:
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            if self.gauss_kernel > 0 and self.max_jitter > 0:
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.ColorJitter(brightness=(0,self.max_jitter), contrast=(0,self.max_jitter), saturation=(0,self.max_jitter), hue=(0,self.max_jitter)),
                    transforms.GaussianBlur(self.gauss_kernel, sigma=(0.1, 2.0)),
                    transforms.RandomHorizontalFlip(p=self.noise),
                    transforms.RandomVerticalFlip(p=self.noise),
                ])
            elif self.gauss_kernel > 0:
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.GaussianBlur(self.gauss_kernel, sigma=(0.1, 2.0)),
                    transforms.RandomHorizontalFlip(p=self.noise),
                    transforms.RandomVerticalFlip(p=self.noise),
                ])
            elif self.max_jitter > 0:
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.ColorJitter(brightness=(0,self.max_jitter), contrast=(0,self.max_jitter), saturation=(0,self.max_jitter), hue=(0,self.max_jitter)),
                    transforms.RandomHorizontalFlip(p=self.noise),
                    transforms.RandomVerticalFlip(p=self.noise),
                ])
            else:
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.RandomHorizontalFlip(p=self.noise),
                    transforms.RandomVerticalFlip(p=self.noise),
                ])
        image = preprocess(image)

        if self.augmentation:
            image_name = image_name.split("_augm_")[0]

        # Labels assignment
        if self.label_type == "tensors":
            label = torch.FloatTensor(np.zeros(self.num_classes))
            classes = self.dataset[image_name]
            for cls in classes:
                label = torch.add(label, self.class_encodings[cls])
                label = torch.where(label > 0, 1.0, 0.0)
        elif self.label_type == "indices":
            label = []
            classes = self.dataset[image_name]
            for cls in classes:
                label.append(self.class_indices[cls])
            label = torch.LongTensor(np.array(label))
        else:
            print("Unsupported label type.")
        
        # Label smoothing
        if self.label_smoothing > 0:
            if self.label_type != "tensors":
                print("Label smoothing is not supported for indices and will be discarded!")
            else:
                positive_classes = torch.sum(label)
                label = torch.where(label > 0, float(1.0 - self.label_smoothing/positive_classes), 0.0)
                complement = torch.where(label == 0, float(self.label_smoothing/(label.shape[0] - positive_classes)), 0.0)
                label = label + complement

        return image, label

    def __len__(self):
        return len(self.row_ids)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def init_ffnn(self, input_dim, embedding_dim, hid_dims=None):
        inp_dim = input_dim
        num_classes = embedding_dim
        if hid_dims is None:
            self.model.classifier = nn.Sequential(
                nn.Linear(inp_dim, num_classes),
                nn.Sigmoid()
            )
        elif len(hid_dims) == 1:
            self.model.classifier = nn.Sequential(
                nn.Linear(inp_dim, hid_dims[0]),
                nn.ReLU(),
                nn.Linear(hid_dims[0], num_classes),
                nn.Sigmoid()
            )
        elif len(hid_dims) == 2:
            self.model.classifier = nn.Sequential(
                nn.Linear(inp_dim, hid_dims[0]),
                nn.ReLU(),
                nn.Linear(hid_dims[0], hid_dims[1]),
                nn.ReLU(),
                nn.Linear(hid_dims[1], num_classes),
                nn.Sigmoid()
            )
        elif len(hid_dims) == 3:
            self.model.classifier = nn.Sequential(
                nn.Linear(inp_dim, hid_dims[0]),
                nn.ReLU(),
                nn.Linear(hid_dims[0], hid_dims[1]),
                nn.ReLU(),
                nn.Linear(hid_dims[1], hid_dims[2]),
                nn.ReLU(),
                nn.Linear(hid_dims[2], num_classes),
                nn.Sigmoid()
            )
        else:
            print("The model can have up to three layers. The remaining layers will be ignored.")
            self.model.classifier = nn.Sequential(
                nn.Linear(inp_dim, hid_dims[0]),
                nn.ReLU(),
                nn.Linear(hid_dims[0], hid_dims[1]),
                nn.ReLU(),
                nn.Linear(hid_dims[1], hid_dims[2]),
                nn.ReLU(),
                nn.Linear(hid_dims[2], num_classes),
                nn.Sigmoid()
            )
            self.model.classifier.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


class Wrapper(nn.Module):
    def __init__(self):
        super(Wrapper, self).__init__()
        self.classifier = None

    def forward(self, x):
        x = self.classifier(x)
        return x


class Tagger(Encoder):
    """
    Tagger with unsupervised training.
    """
    def __init__(self, hid_dim, model=None):
        super(Tagger, self).__init__()
        latent_dim = hid_dim[0] * hid_dim[1]
        self.model = model if not model is None else Wrapper()
        self.init_ffnn(input_dim=latent_dim, embedding_dim=hid_dim[0])

    def forward(self, x):
        x = torch.flatten(x, start_dim=0)
        x = self.model(x)
        return x


class CheXNet(Encoder):
    """
    CheXNet.
    """
    def __init__(self, model, hid_dim, inp_dim=1024):
        super(CheXNet, self).__init__()
        self.model = model
        self.init_ffnn(inp_dim, hid_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class DenseNet121(Encoder):
    """
    DenseNet121 from `"torchvision models" <https://pytorch.org/vision/stable/models.html>`_.
    """
    def __init__(self, hid_dim, inp_dim=1024, pretrained=True):
        super(DenseNet121, self).__init__()
        self.model = models.densenet121(pretrained=pretrained)
        self.init_ffnn(inp_dim, hid_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class DenseNet161(Encoder):
    """
    DenseNet161 from `"torchvision models" <https://pytorch.org/vision/stable/models.html>`_.
    """
    def __init__(self, hid_dim, inp_dim=2208, pretrained=True):
        super(DenseNet161, self).__init__()
        self.model = models.densenet161(pretrained=pretrained)
        self.init_ffnn(inp_dim, hid_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class AlexNet(Encoder):
    """
    AlexNet from `"torchvision models" <https://pytorch.org/vision/stable/models.html>`_.
    """
    def __init__(self, hid_dim, inp_dim=9216, pretrained=True):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained=pretrained)
        self.init_ffnn(inp_dim, hid_dim)

    def forward(self, x):
        x = self.model(x)
        return x

class VGG13(Encoder):
    """
    VGG13 from `"torchvision models" <https://pytorch.org/vision/stable/models.html>`_.
    """
    def __init__(self, hid_dim, inp_dim=25088, pretrained=True):
        super(VGG13, self).__init__()
        self.model = models.vgg13(pretrained=pretrained)
        self.init_ffnn(inp_dim, hid_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class VGG16(Encoder):
    """
    VGG16 from `"torchvision models" <https://pytorch.org/vision/stable/models.html>`_.
    """
    def __init__(self, hid_dim, inp_dim=25088, pretrained=True):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        self.init_ffnn(inp_dim, hid_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet101(Encoder):
    """
    DenseNet121 from `"torchvision models" <https://pytorch.org/vision/stable/models.html>`_.
    """
    def __init__(self, hid_dim, inp_dim=2048, pretrained=True):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(pretrained=pretrained)
        self.init_ffnn(inp_dim, hid_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet50(Encoder):
    """
    DenseNet121 from `"torchvision models" <https://pytorch.org/vision/stable/models.html>`_.
    """
    def __init__(self, hid_dim, inp_dim=2048, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.init_ffnn(inp_dim, hid_dim)

    def forward(self, x):
        x = self.model(x)
        return x



class EfficientNetb0(Encoder):
    """
    EfficientNetb0 from `"torchvision models" <https://pytorch.org/vision/stable/models.html>`_.
    """
    def __init__(self, hid_dim, inp_dim=1280, pretrained=True):
        super(EfficientNetb0, self).__init__()
        self.model = models.efficientnet_b0(pretrained=pretrained)
        self.init_ffnn(inp_dim, hid_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientNetb3(Encoder):
    """
    EfficientNetb3 from `"torchvision models" <https://pytorch.org/vision/stable/models.html>`_.
    """
    def __init__(self, hid_dim, inp_dim=1536, pretrained=True):
        super(EfficientNetb3, self).__init__()
        self.model = models.efficientnet_b3(pretrained=pretrained)
        self.init_ffnn(inp_dim, hid_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientNetb5(Encoder):
    """
    EfficientNetb5 from `"torchvision models" <https://pytorch.org/vision/stable/models.html>`_.
    """
    def __init__(self, hid_dim, inp_dim=9216, pretrained=True):
        super(EfficientNetb5, self).__init__()
        self.model = models.efficientnet_b5(pretrained=pretrained)
        self.init_ffnn(inp_dim, hid_dim)

    def forward(self, x):
        x = self.model(x)
        return x


def label_to_one_hot(unique_tags):
    unique_tags_one_hot = {}
    print("Converting tags to one hot vectors...")
    tags = unique_tags.keys()
    for tag in tqdm(tags):
        tag_vector = np.zeros(len(unique_tags))
        tag_vector[unique_tags[tag]] = 1
        unique_tags_one_hot[tag] = torch.FloatTensor(tag_vector)
    return unique_tags_one_hot


def revert_unique_tags(unique_tags):
    unique_tags_reverse = {}
    print("Reversing unique tags dictionary...")
    tags = unique_tags.keys()
    for tag in tqdm(tags):
        unique_tags_reverse[unique_tags[tag]] = tag
    return unique_tags_reverse


def filenames_test_set(dev_set):
    raw_id = 0
    filenames_IDs = {}
    try:
        filenames = dev_set.keys()
    except:
        filenames = dev_set
    for file in filenames:
        filenames_IDs[raw_id] = file
        raw_id += 1
    return filenames_IDs


def plot_loss(loss_train, loss_val, loss_test=None):
    """ Loss plotting per epoch """
    num_epochs = len(loss_train)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, loss_train, 'g-', label='Training loss')
    plt.plot(epochs, loss_val, 'b-', label='Validation loss')
    plt.plot(epochs, loss_test, 'r-', label='Development loss')
    plt.title('Loss plot per epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc=4)


def plot_F1(f1_train, f1_val, acc_test=None):
    """ F1 plotting per epoch """
    num_epochs = len(f1_train)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, f1_train, 'g-', label='Training F1')
    plt.plot(epochs, f1_val, 'b-', label='Validation F1')
    plt.plot(epochs, acc_test, 'r-', label='Development accuracy')
    plt.title('F1 plot per epoch')
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.legend(loc=4)


def plot_accuracy(acc_train, acc_val, acc_test=None):
    """ Accuracy plotting per epoch """
    num_epochs = len(acc_train)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, acc_train, 'g-', label='Training accuracy')
    plt.plot(epochs, acc_val, 'b-', label='Validation accuracy')
    plt.plot(epochs, acc_test, 'r-', label='Development accuracy')
    plt.title('Accuracy plot per epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc=4)


def generate_plots(loss_train, f1_train, acc_train, loss_validation, f1_validation, acc_validation, loss_development, f1_development, acc_development, directory=None):
    num_plots = 2

    # Plotting loss per epoch
    current_plot = 1
    # plt.subplot(1, num_plots, current_plot)
    # plot_loss(loss_train, loss_validation, loss_development)

    # Plotting accuracy per epoch
    # current_plot += 1
    plt.subplot(1, num_plots, current_plot)
    plot_F1(f1_train, f1_validation, f1_development)

    # Plotting accuracy per epoch
    current_plot += 1
    plt.subplot(1, num_plots, current_plot)
    plot_accuracy(acc_train, acc_validation, acc_development)

    if directory is not None:
        plt.savefig(directory)

    plt.show()