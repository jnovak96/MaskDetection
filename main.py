from PIL import Image as pimage
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import xml.etree.ElementTree as ET
import os, os.path
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.interpolate import make_interp_spline, BSpline

# Stores bounding box information for a mask in an image
class MaskObject:
    def __init__(self, xmin, ymin, xmax, ymax, mask_class):
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)
        self.mask_class = mask_class

    def to_string(self):
        print("{} {} {} {} {}".format(self.mask_class, self.xmin, self.ymin, self.xmax, self.ymax))


# Contains multiple mask objects
class MaskImage:
    def __init__(self, mask_objects, image):
        self.mask_objects = mask_objects
        self.image = image

    def to_string(self):
        for mask_object in self.mask_objects:
            print("Image {}".format(self.image))
            mask_object.to_string()


class NeuralNetwork(nn.Module):
    #NN definition, defining 2 layers of convolution for image processing
    def __init__(self):
        super().__init__()

        # 1st convolution layer: 3 in channels, 6 out channels, kernel size=5
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # 2nd convolution layer: 6 in channels, 16 out channels, kernel size=5
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)
    #definition of steps for the NN
    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MaskDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]) + ".jpg")
        image = pimage.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample


# loads all annotations in the directory
def load_annotations(annotation_path):
    annotation_list = os.listdir(annotation_path)
    loaded_annotations = []
    print(annotation_list)
    for annotation in annotation_list:
        annotation_tree = ET.parse(annotation_path + "/" + annotation)
        loaded_annotations.append(annotation_tree.getroot())
    return loaded_annotations


# parses the annotation XML file for information and saves, returns a mask image object
def check_annotation(root):
    i = 0;
    image = root.find('filename').text
    mask_objects = []
    for elem in root.findall('object'):
        mask_class = elem.find('name').text
        if mask_class == "mask_weared_incorrect":
            mask_class = "without_mask"
        bnd = elem.find('bndbox')
        xmin = bnd.find('xmin').text
        ymin = bnd.find('ymin').text
        xmax = bnd.find('xmax').text
        ymax = bnd.find('ymax').text
        # print("object {}: {} {} {} {} {}".format(i, mask_class, xmin, ymin, xmax, ymax, image))
        mask_object = MaskObject(xmin, ymin, xmax, ymax, mask_class)
        mask_objects.append(mask_object)
        i += 1
    mask_image = MaskImage(mask_objects, image)
    return mask_image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# ============= Main ==============
def main():
    images = "data/images"
    preprocessed_images = "data/images_preprocessed"
    annotations = "data/annotations"
    img_height = 64
    img_width = 64

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    print(torch.cuda.is_available())

    # load all annotations
    annos = load_annotations(annotations)
    # create list to store all mask image objects
    all_masks = []
    # initialize all mask image objects
    for annotation in annos:
        all_masks.append(check_annotation(annotation))

    i = 0

    # crop faces from all mask image objects and store as tensor data for processing

    # define transform, apply image normalization and transform image data to tensor data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # store 2 class names in array
    classes = ["without_mask", "with_mask"]
    run_preprocessing = False
    columns = []
    # load full images
    if run_preprocessing:
        for mask_image in all_masks:
            full_img = pimage.open(images + "/" + mask_image.image).convert("RGB")
            full_img_name = mask_image.image
            # crop to sub-images and
            for face in mask_image.mask_objects:
                row = []
                sub_img = full_img.crop((face.xmin, face.ymin, face.xmax, face.ymax))
                sub_img = sub_img.resize((img_width, img_height))
                img_class = face.mask_class
                row.append(str(i))
                row.append(str(classes.index(img_class)))
                row.append(full_img_name)
                columns.append(row)
                sub_img.save(preprocessed_images + "/" + str(i) + ".jpg")
                i += 1
        csv_data = np.asarray(columns)
        np.savetxt("mask_data.csv", csv_data, delimiter=",", fmt='%s')
        print(str(i) + " images preprocessed")

    mask_dataset = MaskDataset("mask_data.csv", preprocessed_images, transform)

    # split dataset
    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(mask_dataset)
    print(dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    batch_size = 4;

    trainloader = torch.utils.data.DataLoader(mask_dataset, batch_size=batch_size,
                                              num_workers=2, sampler=train_sampler)

    validationloader = torch.utils.data.DataLoader(mask_dataset, batch_size=batch_size,
                                              num_workers=2, sampler=valid_sampler)

    net = NeuralNetwork()
    num_classes = 2
    num_epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    losses = []
    losses_x = []
    training = True
    x = 0
    if training:
        for epoch in range(num_epochs):
            running_loss = 0.0
            print("epoch {}".format(epoch))
            for i, sample in enumerate(trainloader, 0):
                inputs = sample['image']
                labels = sample['label']
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    loss = running_loss / 100
                    losses.append(loss)
                    losses_x.append(x)
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, loss))
                    running_loss = 0.0
                    x += 1

    # save model data
    torch.save(net.state_dict(), "./net_save.pth")

    # plot loss function
    loss_arr_x = np.array(losses_x)
    loss_arr_y = np.array(losses)

    xnew = np.linspace(loss_arr_x.min(), loss_arr_x.max(), 300)
    spl = make_interp_spline(loss_arr_x, loss_arr_y, k=3)
    losses_smooth = spl(xnew)

    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.7])
    ax.set_ylabel('Loss')
    ax.set_xlabel('Batches performed (in hundreds)')
    line, = ax.plot(xnew, losses_smooth, lw=2)
    ax.set_title("Training Loss")
    plt.show()

    dataiter = iter(validationloader)
    data = dataiter.next()
    images = data['image']
    labels = data['label']

    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    net.load_state_dict(torch.load("./net_save.pth"))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in validationloader:
            images = data['image']
            labels = data['label']
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))


if __name__ == "__main__":
    main()
