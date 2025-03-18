# -----------
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import models
from models import CustomNet
import torch.nn as nn

# TRAIN LOOP
def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, total=len(train_loader))):
        inputs, targets = inputs.cuda(), targets.cuda()

        # predictions + loss
        pred = model(inputs)
        loss = criterion(pred, targets)
        # print("=====")
        # print(pred)

        """
        pred and targets do not have the same dimensions. pred has an extra dimension
        representing the probabilities (or logits) for each class, while targets only contains the index of
        the correct class for each image.
        However, the Cross-Entropy Loss function is designed to handle these different dimensions by
        implicitly performing a softmax operation on pred and comparing it to the one-hot encoded representation of targets.

        The input (pred) is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to 1,
        in general). input has to be a Tensor of size (C) for unbatched input, (minibatch, C)
        or (minibatch, C, d1, d2, ..., dK) for the K>=1 dimentional case.
        The last being useful for higher dimension inputs, such as computing cross entropy loss per-pixel for 2D images.
        """

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() # it extracts the value from tensor
        values, predicted_classes = pred.max(1) # it returns the value (max) and the class for each input
        total += targets.size(0)
        correct += predicted_classes.eq(targets).sum().item() # from tensor to scalar (it extracts the value from tensor)
        # print(f'"Batch #{batch_idx}')

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

# VALIDATION LOOP
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            pred = model(inputs)
            loss = criterion(pred, targets)

            val_loss += loss.item()
            values, predicted_classes = pred.max(1)
            total += targets.size(0)
            correct += predicted_classes.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

def main():

  transform = T.Compose([
      T.Resize((100, 100)),  # Resize to fit the input dimensions of the network
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  # root/{class}/x001.jpg

  tiny_imagenet_dataset_train = ImageFolder(root='dataset/tiny-imagenet/tiny-imagenet-200/train', transform=transform)
  tiny_imagenet_dataset_val = ImageFolder(root='dataset/tiny-imagenet/tiny-imagenet-200/val', transform=transform)

  print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
  print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")

  train_loader = DataLoader(tiny_imagenet_dataset_train, batch_size=64, shuffle=True)
  val_loader = DataLoader(tiny_imagenet_dataset_val, batch_size=64, shuffle=False)

  # Model, loss, optimizer
  model = CustomNet.CustomNet().cuda()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

  best_acc = 0

  # Run the training process for {num_epochs} epochs
  num_epochs = 10
  for epoch in range(1, num_epochs + 1):
      print(len(train_loader))
      train(epoch, model, train_loader, criterion, optimizer)


      # At the end of each training iteration, perform a validation step
      val_accuracy = validate(model, val_loader, criterion)

      # Best validation accuracy
      best_acc = max(best_acc, val_accuracy)


  print(f'Best validation accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
  main()