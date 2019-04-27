#from dataloader import get_dataloaders
from daliloader import get_dataloaders
from plot_confusion_matrix import plot_confusion_matrix
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as utils


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_channels, channels, stride, down_sample=lambda x: x):
    super().__init__()
    self.conv1 = nn.Conv2d(
        in_channels, channels, 3, stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(channels)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(channels, channels, 3, 1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(channels)
    self.down_sample = down_sample
    self.relu2 = nn.ReLU()

  def forward(self, x):
    y = self.conv1(x)
    y = self.bn1(y)
    y = self.relu1(y)
    y = self.conv2(y)
    y = self.bn2(y)
    x = self.down_sample(x)
    x = x + y
    x = self.relu2(x)
    return x


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_channels, channels, stride, down_sample=lambda x: x):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, channels, 1, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(channels)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(
        channels, channels, 3, stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(channels)
    self.relu2 = nn.ReLU()
    self.conv3 = nn.Conv2d(channels, channels * 4, 1, 1, bias=False)
    self.bn3 = nn.BatchNorm2d(channels * 4)
    self.down_sample = down_sample
    self.relu3 = nn.ReLU()

  def forward(self, x):
    y = self.conv1(x)
    y = self.bn1(y)
    y = self.relu1(y)
    y = self.conv2(y)
    y = self.bn2(y)
    y = self.relu2(y)
    y = self.conv3(y)
    y = self.bn3(y)
    x = self.down_sample(x)
    x = x + y
    x = self.relu3(x)
    return x


class ResNet(nn.Module):
  def __init__(self, block, layers):
    def get_conv(in_channels, channels, layer, stride):
      expand_channels = channels * block.expansion
      down_sample = lambda x: x
      if stride != 1 or in_channels != expand_channels:
        down_sample = nn.Sequential(
            nn.Conv2d(in_channels, expand_channels, 1, stride, bias=False),
            nn.BatchNorm2d(expand_channels),
        )
      res = [block(in_channels, channels, stride, down_sample)] + [
          block(expand_channels, channels, 1) for _ in range(1, layer)
      ]
      return nn.Sequential(*res)

    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    muls = [1] + [block.expansion] * 3
    channels = [64, 64, 128, 256, 512]
    strides = [1, 2, 2, 2]
    self.convs = nn.ModuleList([
        get_conv(in_ * m, out_, layer,
                 stride) for m, in_, out_, layer, stride in zip(
                     muls, channels[:-1], channels[1:], layers, strides)
    ])
    self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.flat_dim = 512 * block.expansion
    self.classify = nn.Linear(in_features=self.flat_dim, out_features=5)

  def forward(self, x):
    x = self.conv1(x)
    for conv in self.convs:
      x = conv(x)
    x = self.avg_pool(x)
    x = x.view(-1, self.flat_dim)
    x = self.classify(x)
    return x

def load_pretrained(name):
  from torchvision import models
  nets = {
      'ResNet18': models.resnet18,
      'ResNet50': models.resnet50,
  }
  net = nets[name](pretrained=True)
  #for param in net.parameters():
  #  param.requires_grad = False
  net.fc = nn.Linear(net.fc.in_features, 5)
  return net


def main(args):
  net = None
  if args.pretrained:
    net = load_pretrained(args.net).to(args.device)
  else:
    net = Nets[args.net].to(args.device)
  #print(net)
  criterion = nn.CrossEntropyLoss()
  #optim = torch.optim.Adam
  optim = torch.optim.SGD
  optimizer = optim(
      net.parameters(),
      lr=args.lr,
      momentum=0.9,
      #nesterov=True,
      weight_decay=args.weight_decay)
  train_loader, test_loader = get_dataloaders(args.batch_size, args.device)
  epoch_start = 1

  # restore model
  if args.restore:
    print('> Restore from', args.path)
    checkpoint = torch.load(args.path)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch']
    if 'batch' in checkpoint:
      print('> Start from [{:2d}.{:04d}] with batch accuracy {:.2%}'.format(
          epoch_start, checkpoint['batch'], checkpoint['train_acc']))
    else:
      print(
          '> Start from [{:2d}] with (train, test) accuracy ({:.2%}, {:.2%})'.
          format(epoch_start, checkpoint['train_acc'], checkpoint['test_acc']))

  for epoch in range(epoch_start, args.epochs + 1):
    correct, total = 0, 0
    #for i, (inputs, labels) in enumerate(train_loader, start=1):
    #  labels = labels.view(-1)
    train_loader.reset()
    test_loader.reset()
    for i, data in enumerate(train_loader, start=1):
      inputs = data[0]['data']
      labels = data[0]['label'].squeeze().cuda().long()

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      logits = net(inputs)
      loss = criterion(logits, labels)
      loss.backward()
      optimizer.step()

      # batch train accuracy
      with torch.no_grad():
        predicted = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
      print('[{:2d}.{:04d}] acc: {:.2%}'.format(epoch, i, correct / total))

      # model checkpoint
      #if i % 100 == 0:
      if False:
        torch.save({
            'epoch': epoch,
            'batch': i,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_acc': correct / total,
        }, f'{args.path}.ckpt')

    # train accuracy
    train_acc = correct / total

    # test accuracy
    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
      #for i, (inputs, labels) in enumerate(test_loader):
      #  labels = labels.view(-1)
      for i, data in enumerate(test_loader):
        inputs = data[0]['data']
        labels = data[0]['label'].squeeze().cuda().long()

        logits = net(inputs)
        predicted = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_acc = correct / total
    print('[{:2d}] train acc: {:.2%} test acc: {:.2%}'.format(
        epoch, train_acc, test_acc))

    # save model
    torch.save({
        'epoch': epoch,
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_acc': train_acc,
        'test_acc': test_acc,
    }, f'{args.path}.{epoch}')

  # confusion matrix
  print('> Plot Confusion Matrix')
  net.eval()
  y_true, y_pred = np.array([]), np.array([])
  #test_len = len(test_loader)
  test_len = test_loader._size // args.batch_size
  with torch.no_grad():
    #for i, (inputs, labels) in enumerate(test_loader):
    #  labels = labels.view(-1)
    test_loader.reset()
    for i, data in enumerate(test_loader):
      inputs = data[0]['data']
      labels = data[0]['label'].squeeze().cuda().long()

      logits = net(inputs)
      predicted = torch.argmax(logits, dim=1)
      y_true = np.append(y_true, labels.cpu())
      y_pred = np.append(y_pred, predicted.cpu())
      print('{:.2%} ({}/{})'.format(i / test_len, i, test_len))
  plot_confusion_matrix(y_true, y_pred, classes=np.arange(5))


Nets = {
    'ResNet18': ResNet(BasicBlock, [2, 2, 2, 2]),
    'ResNet50': ResNet(Bottleneck, [3, 4, 6, 3]),
}

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # network
  parser.add_argument('-n', '--net', default='ResNet18', choices=Nets.keys())
  parser.add_argument('-p', '--path', default='model/model.pth')
  parser.add_argument('-r', '--restore', action='store_true')
  parser.add_argument('-pt', '--pretrained', action='store_true')
  # training
  parser.add_argument('-d', '--device', default='cuda')
  parser.add_argument('-bs', '--batch_size', default=64, type=int)
  parser.add_argument('-e', '--epochs', default=10, type=int)
  parser.add_argument('-lr', '--lr', default=0.001, type=float)
  parser.add_argument('-wd', '--weight_decay', default=0, type=float)

  args = parser.parse_args()
  main(args)
