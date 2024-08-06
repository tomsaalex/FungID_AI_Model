import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io
from torch import nn, softmax
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

import timm
from MO_106_Dataset import MO_106_Dataset
from torch.utils.tensorboard import SummaryWriter

from m_vit.m_vit import MVitClassifier

weight_decay = 5e-3
learning_rate = 5e-4
batch_size = 32
epochs = 300
shuffle_dataset = True
random_seed = 42
validation_split = .2

mo106_mean = 274.528301886792
mo106_std = 61.6721296237372

def train():
    os.makedirs("model_saves", exist_ok=True)
    mushroom_dataset = MO_106_Dataset("data/mo106_dataset.csv", "data",
                                      transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation((45, 90)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mo106_mean, std=mo106_std)
                                      ]))

    dataset_size = len(mushroom_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(mushroom_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_dataloader = DataLoader(mushroom_dataset, batch_size=batch_size, sampler=valid_sampler)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    #model = timm.create_model("mobilevit_s", pretrained=False, num_classes=106)
    model = MVitClassifier(3, 106, batch_size)
    model.to(device)
    writer = SummaryWriter()

    print(model)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    global_step = 0

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------------")
        if t % 5 == 0 or t == epochs - 1:
            torch.save(model.state_dict(), f"model_saves/model_weights_epoch_{t}.pth")
            print(f"Model saved at epoch {t}.")

        global_step = train_loop(train_dataloader, device, model, loss_fn, optimizer, writer, global_step)
        validation_loop(validation_dataloader, device, model, loss_fn, writer, t+1)

    writer.flush()
    writer.close()
    print("Done!")


def train_loop(dataloader, device, model, loss_fn, optimizer, writer, global_step):
    size = len(dataloader.sampler.indices)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        print(f"Starting batch: {batch}")
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        writer.add_scalar("Loss/train", loss, global_step)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)

            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return global_step


def validation_loop(dataloader, device, model, loss_fn, writer, epoch_num):
    size = len(dataloader.sampler.indices)
    num_batches = len(dataloader)
    validation_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            validation_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    validation_loss /= num_batches
    correct /= size

    writer.add_scalar("Loss/validation", validation_loss, epoch_num)
    writer.add_scalar("Accuracy/validation", 100*correct, epoch_num)
    print(f"Validation: \n Accuracy: {100 * correct:>0.1f}%, Avg. loss: {validation_loss:>8f} \n")


def eval():
    mushroom_dataset = MO_106_Dataset("data/mo106_dataset.csv", "data",
                                      transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.ToTensor()
                                      ]))

    model = timm.create_model("mobilevitv2_200", pretrained=True, num_classes=106)
    model.load_state_dict(torch.load("model_saves/second_run_106_classes/model_weights_epoch_35.pth"))
    model.eval()

    image = io.imread("random_shrooms/Shroom_10.jpg")
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    image = transform_pipeline(image)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()

    # Add a batch dimension
    image = image.unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        prediction = model(image)
        # prediction = torch.argmax(prediction, dim=1)
        prediction = softmax(prediction, dim=1)
        predictions_list = prediction.tolist()[0]
        print(predictions_list)
        print(max(predictions_list))
        # print the index of the class with the highest probability
        print(f"Predicted class: {np.argmax(predictions_list)}")
    print(prediction)
    # print(prediction.shape)


def run():
    train()
    # eval()
    # ba = BlockAttention(2, 3, 8, 8, 4)
    # tensor = torch.tensor([
    #     [
    #         [
    #             [1., 1., 1., 1., 1., 1., 1., 1.],
    #             [1., 1., 1., 1., 1., 1., 1., 1.],
    #             [1., 1., 1., 1., 1., 1., 1., 1.],
    #             [1., 1., 1., 1., 1., 1., 1., 1.],
    #             [1., 1., 1., 1., 1., 1., 1., 1.],
    #             [1., 1., 1., 1., 1., 1., 1., 1.],
    #             [1., 1., 1., 1., 1., 1., 1., 1.],
    #             [1., 1., 1., 1., 1., 1., 1., 1.],
    #         ]
    #         ,
    #         [
    #             [2., 2., 2., 2., 2., 2., 2., 2.],
    #             [2., 2., 2., 2., 2., 2., 2., 2.],
    #             [2., 2., 2., 2., 2., 2., 2., 2.],
    #             [2., 2., 2., 2., 2., 2., 2., 2.],
    #             [2., 2., 2., 2., 2., 2., 2., 2.],
    #             [2., 2., 2., 2., 2., 2., 2., 2.],
    #             [2., 2., 2., 2., 2., 2., 2., 2.],
    #             [2., 2., 2., 2., 2., 2., 2., 2.],
    #         ],
    #         [
    #             [3., 3., 3., 3., 3., 3., 3., 3.],
    #             [3., 3., 3., 3., 3., 3., 3., 3.],
    #             [3., 3., 3., 3., 3., 3., 3., 3.],
    #             [3., 3., 3., 3., 3., 3., 3., 3.],
    #             [3., 3., 3., 3., 3., 3., 3., 3.],
    #             [3., 3., 3., 3., 3., 3., 3., 3.],
    #             [3., 3., 3., 3., 3., 3., 3., 3.],
    #             [3., 3., 3., 3., 3., 3., 3., 3.],
    #         ]],
    #     [
    #         [
    #             [4., 4., 4., 4., 4., 4., 4., 4.],
    #             [4., 4., 4., 4., 4., 4., 4., 4.],
    #             [4., 4., 4., 4., 4., 4., 4., 4.],
    #             [4., 4., 4., 4., 4., 4., 4., 4.],
    #             [4., 4., 4., 4., 4., 4., 4., 4.],
    #             [4., 4., 4., 4., 4., 4., 4., 4.],
    #             [4., 4., 4., 4., 4., 4., 4., 4.],
    #             [4., 4., 4., 4., 4., 4., 4., 4.],
    #         ]
    #         ,
    #         [
    #             [5., 5., 5., 5., 5., 5., 5., 5.],
    #             [5., 5., 5., 5., 5., 5., 5., 5.],
    #             [5., 5., 5., 5., 5., 5., 5., 5.],
    #             [5., 5., 5., 5., 5., 5., 5., 5.],
    #             [5., 5., 5., 5., 5., 5., 5., 5.],
    #             [5., 5., 5., 5., 5., 5., 5., 5.],
    #             [5., 5., 5., 5., 5., 5., 5., 5.],
    #             [5., 5., 5., 5., 5., 5., 5., 5.],
    #         ],
    #         [
    #             [6., 6., 6., 6., 6., 6., 6., 6.],
    #             [6., 6., 6., 6., 6., 6., 6., 6.],
    #             [6., 6., 6., 6., 6., 6., 6., 6.],
    #             [6., 6., 6., 6., 6., 6., 6., 6.],
    #             [6., 6., 6., 6., 6., 6., 6., 6.],
    #             [6., 6., 6., 6., 6., 6., 6., 6.],
    #             [6., 6., 6., 6., 6., 6., 6., 6.],
    #             [6., 6., 6., 6., 6., 6., 6., 6.],
    #         ]]
    # ])
    # torch.set_printoptions(threshold=10_000)
    # print(tensor)
    # # print(tensor.shape)
    #
    # windows = ba.split_into_windows(tensor)
    # print(windows)
    # print(windows.shape)
    #
    # print(ba.combine_windows(windows))


if __name__ == '__main__':
    print("Running")
    run()
