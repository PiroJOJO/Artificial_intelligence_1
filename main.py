import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from tqdm import tqdm
from torchvision import datasets, models, transforms
from torch.nn import functional as F


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

data = datasets.ImageFolder(root='archive-2/simpsons_dataset', transform=data_transforms)
data_test = datasets.ImageFolder(root='archive-2/test', transform=data_transforms)
generator = torch.Generator().manual_seed(42)
data_train, data_valid = torch.utils.data.random_split(data, [0.9, 0.1], generator=generator)
data_image = {
    'train':
        data_train,
    'validation':
        data_valid,
    'test':
        data_test
}
classes_image = list(data.class_to_idx.keys())
loaders_image = {
    'train':
        torch.utils.data.DataLoader(data_image['train'], batch_size=32, shuffle=True),
    'validation':
        torch.utils.data.DataLoader(data_image['validation'], batch_size=32, shuffle=True),
    'test':
        torch.utils.data.DataLoader(data_image['test'], batch_size=32, shuffle=True)
}
# path_for_test = 'archive-2/kaggle_simpson_testset/kaggle_simpson_testset'
# names = []
# for dirs, folder, files in os.walk(path_for_test):
#     for img in files:
#         ind = '_'.join(os.path.splitext(os.path.basename(path_for_test + '/' + img))[0].split('_')[:-1])
#         names.append(ind)
# names = list(set(names))
# print(len(names))
# folder_path = './archive-2'
# os.mkdir(folder_path+'/test')
# folder_path = folder_path + '/test'
# for folder in names:
#     if not os.path.exists(folder_path + '/' + folder):
#         os.mkdir(folder_path + '/' + folder)
# list_of_names = os.listdir(folder_path)
# print(list_of_names)
# for dirs, folder, files in os.walk(path_for_test):
#     for img in files:
#         ind = '_'.join(os.path.splitext(os.path.basename(path_for_test + '/' + img))[0].split('_')[:-1])
#         for name in list_of_names:
#             if ind == name:
#                 shutil.move(path_for_test + '/' + img, folder_path + '/' + name)

def image_shower(images, labels, n=4):

    fig, axes = plt.subplots(1, 4)
    fig.set_figwidth(12)  # ширина и
    fig.set_figheight(6)
    for i, image in enumerate(images[:n]):
        # plt.subplot(n, n, i + 1)
        image = image * 0.2 + 0.4
        axes[i].imshow(image.numpy().transpose((1, 2, 0)).squeeze())
        axes[i].set_title(classes_image[labels[i]])

        # plt.imshow(image.numpy().transpose((1, 2, 0)).squeeze())
    plt.show()
    print("Real Labels: ", ' '.join('%5s' % classes_image[label] for label in labels[:n]))

images, labels = next(iter(loaders_image['train']))

# image_shower('train', images, labels)
image_shower( images, labels)
# data_test = datasets.ImageFolder(root ='/content/dataset/archive-2/kaggle_simpson_testset',
#                                          transform = data_transforms)
# print(data_test)
# loaders_test = torch.utils.data.DataLoader(data_test, batch_size=32, shuffle=True, num_workers=0)
# print(loaders_test)
# names_train = []
# for i, (images, labels) in enumerate(data_train):
#   names_train.append(labels)
# print(names_train)

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 42))
# img = torch.from_numpy(np.zeros((3, 224, 224))).float()
# # Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)

# # Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
# print(prediction)
# print(prediction.shape)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())


def train_model(model, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loaders_image[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_image[phase])
            epoch_acc = running_corrects.double() / len(data_image[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
    return model
def predict(model):
    outputs = model(data_image['test'])
    _, preds = torch.max(outputs, 1)
    return preds

def to_names(y):
    result = []
    for i in y:
      result.append(classes_image['test'][i])
      return result

#посчитаем точность работы нашего классификатора
correct = 0
total = 0
with torch.no_grad():
    #переводим модель в режим инференса
    model.eval()
    for data in data_image['test']:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        #получаем наши предсказания
        _, predicted = torch.max(outputs.data, 1)
        #посчитаем общее количество картинок
        total += labels.size(0)
        #посчитаем количество точно классифицированных картинок
        correct += (predicted == labels).sum().item()
print("Accuracy: %d" %(100 * correct/total))