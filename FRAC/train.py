import torch
import torchvision
import torchvision.transforms as transforms
from monai.visualize import GradCAM
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image

transform = transforms.Compose(
    [transforms.ToTensor()
     ])

dataset = torchvision.datasets.ImageFolder(root='', transform=transform)


train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                          shuffle=False, num_workers=2)


classes = dataset.classes

model = torchvision.models.densenet201(pretrained=True)


# for param in model.parameters():
#     param.requires_grad = False


# model.fc = torch.nn.Linear(model.fc.in_features, 2)


num_features = model.classifier.in_features
model.classifier = torch.nn.Linear(num_features, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# cam = grad_cam(model=model, layer="layer2") 

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epochs = 1000
best_acc = 0.0
for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        probabilities = torch.softmax(outputs, dim=1)
        engine_coeff = probabilities[:, 1] # engine_coeff is the probability of the image being a fine label
        loss.backward()
        optimizer.step()

        # cam.get_cam(inputs=inputs)

        print('epoch: %d, loss: %.3f' % (epoch + 1, loss.item()))



        running_loss += loss.item()
        if i % 200 == 199:
            print('average loss: [%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0


    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    if accuracy > best_acc:
        best_acc = accuracy

        torch.save(model, './FRAC/pretrain_model/densenet-201-best-model-{}.pth'.format(accuracy))

        # torch.save(model.state_dict(), 'best-model-parameters-{}.pth'.format(accuracy))

    print('Accuracy of the network on the test images: %d %%' % accuracy)

print('Finished Training')
