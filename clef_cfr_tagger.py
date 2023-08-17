import cnn_backbone as backbone
import imageCLEF_dataset as splits
import imageCLEF23_dataset as dataset
import cgen_data_loader as dataloader
import clef_rnn_captioning as model
import bert_backbone as encoder
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import numpy as np
import os
import re

# Hyperparameters
USE_CLS = False
PRECOMPUTED_DOC_EMB = True
LEARNING_RATE = 10e-3
LEARNING_RATE_DECAY = "linear"
NUM_EPOCHS = 20
TRAIN_BATCH = 5
VAL_BATCH = 5
TEST_BATCH = 5

MODELS = ["cambridgeltl/BioRedditBERT-uncased", "dmis-lab/biobert-large-cased-v1.1-squad", "dmis-lab/TinyPubMedBERT-v1.0"]

# Helper functions
def create_target(caption, tok2idx):
    not_found = []
    counter = 0
    targets = None
    for token in caption.split(" "):
        if token in tok2idx.keys():
            targets = tok2idx[token] if targets is None \
            else torch.cat((targets, tok2idx[token]))
        else:
            not_found.append(token)
        counter += 1

    return targets, counter, not_found

# Create the dataset folder, download (tagging) data and split to train and test
# img_clef_23 = "/media/geomos/AUEB BIOMEDICAL SYSTEMS/ImageCLEF2023/imageCLEF2023"
img_clef_23 = "/media/georg_mosh/Samsung_T5/AUEB BIOMEDICAL SYSTEMS/imageCLEF2023"
_, _, _, unique_tags = splits.get_imageclef(val_perc=0.1, dev_perc=0.1, year=2023, base=img_clef_23,
                                            task='classification')
train_set, val_set, dev_set = splits.get_pkaliosis_splits()

# Create the dataset folder, download (captioning) data and split to train and test
data_elements = dataset.create_dataset()
train_capt, val_capt, dev_capt = splits.get_captioning_splits(data_elements.captions_data, train_set, val_set, dev_set)

# Define the folder in which to save the results
results_path = os.path.join(img_clef_23, "imageclef2023_cap_results")

# Define the sub folder that contains the images
images_path = os.path.join(img_clef_23, "images/dataset_github/ImageCLEFmedical_Caption_2023_train_val_images")

# Convert the unique tags to one hot vectors
unique_tags_one_hot = backbone.label_to_one_hot(unique_tags)
unique_tags_cuis = backbone.revert_unique_tags(unique_tags)
unique_tags_diseases = splits.get_diseases(img_clef_23)

# Enable graphics hardware acceleration
device_document_encoder = 'cuda:0'
device_query_encoder = 'cuda:0'
device_tagger = 'cuda:0'
device_generator = 'cuda:0'

if not PRECOMPUTED_DOC_EMB:
    # Create the document encoder and move to the appropriate device
    document_encoder = encoder.HfBertModel(CLS_representation=USE_CLS, device=device_document_encoder,
                                           weights_name=MODELS[2])
    document_encoder = document_encoder.to(device_document_encoder)

    # Compute the concepts embeddings using the LLM
    tags = list(backbone.revert_unique_tags(unique_tags_diseases).keys())
    if not USE_CLS:
        all_tags_embeddings = document_encoder.batch_score_split([tags[0]]).detach().cpu().unsqueeze(0)
        for i in range(1, 2125):
            all_tags_embeddings = torch.cat((all_tags_embeddings,
                                             document_encoder.batch_score_split([tags[i]]).detach().cpu().unsqueeze(0)))
    else:
        all_tags_embeddings = document_encoder.batch_score_split([tags[0]]).detach().cpu()
        for i in range(1, 2125):
            all_tags_embeddings = torch.cat((all_tags_embeddings,
                                             document_encoder.batch_score_split([tags[i]]).detach().cpu()))
    all_tags_embeddings = torch.FloatTensor(all_tags_embeddings)
    all_tags_embeddings = all_tags_embeddings.to(device_tagger)
else:
    # Load the concepts embeddings
    if not USE_CLS:
        all_tags_embeddings = torch.load("./pooled_tags_embeddings_tiny.pt", map_location='cpu')
    else:
        all_tags_embeddings = torch.load("./cls_tags_embeddings_tiny.pt", map_location='cpu')
all_tags_embeddings = all_tags_embeddings.to(device_tagger)

# Create the query encoder and move to the appropriate device
query_encoder = encoder.HfBertQueryModel(CLS_representation=False, device=device_query_encoder,
                                         weights_name=MODELS[2])
query_encoder = query_encoder.to(device_query_encoder)

# Create the classification head ('tagger') and move to the appropriate device
tagger = backbone.Tagger(hid_dim=[2125, 312])
tagger = tagger.to(device_tagger)

# Create the LLM (GRU) and move to the appropriate device
generator = model.RecurrentNeuralNetwork(input_size=1280+2125+312, hidden_size=len(data_elements.word2vec), n_layers=1,
                                         device=device_generator)
generator = generator.to(device_generator)

# Set the loss function and the optimizer
criterion = nn.CrossEntropyLoss()

# Set the model optimizers
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
tagger_optimizer = torch.optim.Adam(tagger.parameters(), lr=LEARNING_RATE)

# Define Dataset instances for the train, validation, and development sets
train_data = dataloader.DataLoader2(train_capt, data_elements.image_vectors, "train set")
val_data = dataloader.DataLoader2(val_capt, data_elements.image_vectors, "validation set")
dev_data = dataloader.DataLoader2(dev_capt, data_elements.image_vectors, "development set")

# Define data loaders for the train, validation, and development sets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=VAL_BATCH, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=TEST_BATCH, shuffle=False)

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPOCHS = trange(NUM_EPOCHS, desc='Epoch: ', leave=True)
for epoch in EPOCHS:  # loop over the dataset multiple times
    tagger.train()
    generator.train()
    query_encoder.eval()

    running_train_loss = 0.0
    running_train_accuracy = 0.0
    cumulative_train_F1 = 0.0

    if LEARNING_RATE_DECAY == "linear":
        for g in tagger_optimizer.param_groups:
            g['lr'] = LEARNING_RATE * (1 - epoch / NUM_EPOCHS)

        for g in generator_optimizer.param_groups:
            g['lr'] = LEARNING_RATE * (1 - epoch / NUM_EPOCHS)

    for i, (image, caption) in enumerate(train_loader, 0):
        # zero the parameter gradients
        tagger_optimizer.zero_grad()
        generator_optimizer.zero_grad()

        # get the captions' tokens
        captions = [caption[i].split(" ") for i in range(TRAIN_BATCH)]
        captions_lengths = np.array([len(captions[i]) for i in range(TRAIN_BATCH)])
        max_length = np.max(captions_lengths)

        # apply padding to have the same caption lengths
        for i in range(TRAIN_BATCH):
            for _ in range(len(captions[i]), max_length):
                captions[i].append("<pad>")

        # next token prediction: predict the tokens probabilities
        with torch.no_grad():
            token_embeddings = query_encoder.batch_score_split(captions[0]).unsqueeze(1)
            token_target_embeddings = dataloader.create_targets(re.sub(r"[^A-Za-z0-9 ]+", '', (" ").join(captions[0])).lower().split(" "),
                                                                data_elements.word2vec).unsqueeze(1)
            for i in range(1, TRAIN_BATCH):
                token_embeddings = torch.cat((token_embeddings, query_encoder.batch_score_split(captions[i]).unsqueeze(1)), dim=1)
                token_target_embeddings = torch.cat((token_target_embeddings, dataloader.create_targets(
                                                    re.sub(r"[^A-Za-z0-9 ]+", '', (" ").join(captions[i])).lower().split(" "),
                                                    data_elements.word2vec).unsqueeze(1)), dim=1)
        token_embeddings = token_embeddings.to(device_generator)
        token_target_embeddings = token_target_embeddings.to(device_generator)


        # compute the concepts probabilities (tags unsupervised confidence scores)
        tags_probabilities = tagger(all_tags_embeddings.detach().squeeze(1))
        tags_probabilities = tags_probabilities.repeat(token_embeddings.shape[0], TRAIN_BATCH, 1)
        tags_probabilities = tags_probabilities.to(device_generator)

        # move images to the appropriate device if hardware acceleration can be applied
        image_embeddings = image[0].repeat(token_embeddings.shape[0], 1, 1)
        for i in range(1, TRAIN_BATCH):
            image_embeddings = torch.cat((image_embeddings, image[1].repeat(token_embeddings.shape[0], 1, 1)), dim=1)
        image_embeddings = image_embeddings.to(device_generator)

        # compute the tokens probabilities (via the generative model)
        joint_embedding = torch.cat((image_embeddings, tags_probabilities, token_embeddings), dim=2)

        # apply the next token prediction with teacher forcing
        prediction = generator(joint_embedding)[0]

        # Computing loss function
        loss = criterion(prediction, token_target_embeddings)  # Binary Cross-Entropy

        # Computing gradient
        loss.backward()

        # Performing backward pass (backpropagation)
        generator_optimizer.step()
        tagger_optimizer.step()

print("Extraction completed")
