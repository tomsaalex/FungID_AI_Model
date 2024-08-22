import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io
from torch import nn, softmax
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, \
    ConfusionMatrixDisplay

import timm
from MO_106_Dataset import MO_106_Dataset
from torch.utils.tensorboard import SummaryWriter

from m_vit.m_vit import MVitClassifier

weight_decay = 5e-3
learning_rate = 5e-4
batch_size = 32
epochs = 100
shuffle_dataset = True
random_seed = 42
validation_split = .2

def train():
    os.makedirs("model_saves", exist_ok=True)
    mushroom_dataset = MO_106_Dataset("data/mo106_dataset.csv", "data",
                                      transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation((45, 90)),
                                          transforms.ToTensor()
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
    model = timm.create_model("mobilevitv2_200", pretrained=False, num_classes=106)
    #model = MVitClassifier(3, 106, batch_size)
    model.to(device)
    writer = SummaryWriter()

    print(model)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    global_step = 0

    for t in range(1, epochs + 1):
        print(f"Epoch {t}\n-------------------------------------")
        if t % 5 == 0 or t == epochs:
            torch.save(model.state_dict(), f"model_saves/model_weights_epoch_{t}.pth")
            print(f"Model saved at epoch {t}.")

        global_step = train_loop(train_dataloader, device, model, loss_fn, optimizer, writer, global_step, epoch_num=t)
        validation_loop(validation_dataloader, device, model, loss_fn, writer, t)

    writer.flush()
    writer.close()
    print("Done!")


def train_loop(dataloader, device, model, loss_fn, optimizer, writer, global_step, epoch_num):
    size = len(dataloader.sampler.indices)
    all_preds = []
    all_labels = []

    correctly_classified_samples = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        print(f"Starting batch: {batch}")
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        correctly_classified_samples += (pred.argmax(1) == y).type(torch.float).sum().item()

        all_preds.extend(pred.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        writer.add_scalar("Loss/train", loss, global_step)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)

            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    writer.add_scalar("Accuracy/train", accuracy, epoch_num)
    writer.add_scalar("Precision/train", precision, epoch_num)
    writer.add_scalar("Recall/train", recall, epoch_num)
    writer.add_scalar("F1/train", f1, epoch_num)

    return global_step


def validation_loop(dataloader, device, model, loss_fn, writer, epoch_num):
    size = len(dataloader.sampler.indices)
    num_batches = len(dataloader)
    validation_loss, correctly_classified_samples = 0, 0

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            validation_loss += loss_fn(pred, y).item()
            correctly_classified_samples += (pred.argmax(1) == y).type(torch.float).sum().item()

            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    validation_loss /= num_batches
    accuracy_custom = correctly_classified_samples / size

    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    writer.add_scalar("Loss/validation", validation_loss, epoch_num)
    writer.add_scalar("Accuracy/validation", accuracy, epoch_num)
    writer.add_scalar("Precision/validation", precision, epoch_num)
    writer.add_scalar("Recall/validation", recall, epoch_num)
    writer.add_scalar("F1/validation", f1, epoch_num)
    print(f"Validation: \n Accuracy Custom: {100 * accuracy_custom:>0.5f}%, Avg. loss: {validation_loss:>8f} \n")
    print(f"Accuracy: {accuracy:>0.5f}, Precision: {precision:>0.5}, Recall: {recall:>0.5f}, F1: {f1:>0.5f}")

    # Display Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for Epoch {epoch_num}')
    plt.show()


def eval():
    mushroom_dataset = MO_106_Dataset("data/mo106_dataset.csv", "data",
                                      transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
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
