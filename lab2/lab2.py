from dataloader import read_bci_data
import argparse
import torch
import torch.nn as nn
import torch.utils.data as utils
from torch.nn import functional as F

Activations = nn.ModuleDict([
    ['relu', nn.ReLU()],
    ['lrelu', nn.LeakyReLU()],
    ['elu', nn.ELU()],
])


class Conv2dSame(nn.Conv2d):
  def forward(self, input):
    _, _, *input_size = input.shape
    output_size = [(i + s - 1) // s for i, s in zip(input_size, self.stride)]
    pad = [
        max(0, (o - 1) * s + (k - 1) * d + 1 - i)
        for i, o, k, s, d in zip(input_size, output_size, self.kernel_size,
                                 self.stride, self.dilation)
    ]
    odd = [p % 2 != 0 for p in pad]
    if any(odd):
      input = F.pad(input, [0, int(odd[1]), 0, int(odd[0])])
    padding = [p // 2 for p in pad]
    return F.conv2d(input, self.weight, self.bias, self.stride, padding,
                    self.dilation, self.groups)


class GaussianNoise(nn.Module):
  def __init__(self, sigma=0.1, is_relative_detach=True):
    super().__init__()
    self.sigma = sigma
    self.is_relative_detach = is_relative_detach
    self.noise = torch.tensor(0).to('cuda')

  def forward(self, x):
    if self.training and self.sigma != 0:
      scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
      sampled_noise = self.noise.repeat(*x.size()).normal_() * scale


class DeepConvNet(nn.Module):
  def __init__(self, act):
    super().__init__()
    p = 0.8
    bn_args = {
        #'eps': 1e-5,
        #'momentum': 0.1,
        #'eps': 1e-2,
        #'momentum': 0.99,
        #'track_running_stats': True,
        'momentum': None,
        'track_running_stats': False,
    }
    self.flat_dim = 8600
    conv_kernel = (1, 5)
    pool_kernel, pool_stride = (1, 2), (1, 2)
    #self.flat_dim = 800
    #conv_kernel = (1, 11)
    #pool_kernel, pool_stride = (1, 3), (1, 3)
    self.conv0 = nn.Sequential(
        nn.Conv2d(1, 25, kernel_size=conv_kernel),
        nn.Conv2d(25, 25, kernel_size=(2, 1)),
        nn.BatchNorm2d(25, **bn_args),
        Activations[act],
        nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
        nn.Dropout(p),
    )
    channels = [25, 50, 100, 200]
    self.convs = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel),
            nn.BatchNorm2d(out_channels, **bn_args),
            Activations[act],
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            nn.Dropout(p),
        ) for in_channels, out_channels in zip(channels[:-1], channels[1:])
    ])
    self.classify = nn.Sequential(
        nn.Linear(in_features=self.flat_dim, out_features=2, bias=True))

  def forward(self, x):
    x = self.conv0(x)
    for conv in self.convs:
      x = conv(x)
    x = x.view(-1, self.flat_dim)
    x = self.classify(x)
    return x


class EEGNet(nn.Module):
  def __init__(self, act, ks=51, p=0.25, flat_dim=736, F1=16, D=2, F2=None):
    super().__init__()
    if F2 is None:
      F2 = D * F1
    bn_args = {
        'momentum': None,
        'track_running_stats': False,
    }
    self.flat_dim = flat_dim
    self.firstConv = nn.Sequential(
        Conv2dSame(1, F1, kernel_size=(1, ks), padding=None, bias=False),
        nn.BatchNorm2d(F1, **bn_args),
    )
    self.depthwiseConv = nn.Sequential(
        nn.Conv2d(F1, D * F1, kernel_size=(2, 1), groups=F1, bias=False),
        nn.BatchNorm2d(D * F1, **bn_args),
        Activations[act],
        nn.AvgPool2d((1, 4)),
        nn.Dropout(p),
    )
    self.separableConv = nn.Sequential(
        Conv2dSame(D * F1, F2, kernel_size=(1, 16), padding=None, bias=False),
        nn.BatchNorm2d(F2, **bn_args),
        Activations[act],
        nn.AvgPool2d((1, 8)),
        nn.Dropout(p),
    )
    self.classify = nn.Sequential(
        nn.Linear(in_features=self.flat_dim, out_features=2, bias=True))

  def forward(self, x):
    x = self.firstConv(x)
    x = self.depthwiseConv(x)
    x = self.separableConv(x)
    x = x.view(-1, self.flat_dim)
    x = self.classify(x)
    return x


def get_dataloader(batch_size, device):
  def get_loader(data, label):
    data = torch.stack([torch.Tensor(i) for i in data]).to(device)
    label = torch.LongTensor(label).to(device)
    dataset = utils.TensorDataset(data, label)
    loader = utils.DataLoader(dataset, batch_size)
    return loader

  def preprocess(data):
    import numpy as np
    data = (data - np.mean(data)) / np.std(data)
    data = (2 * data - 1) / 10
    return data

  train_data, train_label, test_data, test_label = read_bci_data()
  # train_data, test_data = preprocess(train_data), preprocess(test_data)
  train_loader = get_loader(train_data, train_label)
  test_loader = get_loader(test_data, test_label)
  return train_loader, test_loader


def main(args):
  arg = [args.activation]
  if args.net == 'EEGNet':
    arg += [args.kernel_size, args.drop_prob, args.flat_dim, args.F1, args.D]
  net = Nets[args.net](*arg).to(args.device)
  # print(net)
  criterion = nn.CrossEntropyLoss()
  # opt = torch.optim.RMSprop
  optim = torch.optim.Adam
  optimizer = optim(
      net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  train_loader, test_loader = get_dataloader(args.batch_size, args.device)
  softmax = nn.Softmax(dim=1)

  for epoch in range(args.epochs):
    correct, total = 0, 0
    for i, (inputs, labels) in enumerate(train_loader, start=1):
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # batch train accuracy
      with torch.no_grad():
        predicted = torch.argmax(softmax(outputs), dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # train accuracy
    train_acc = correct / total

    # test accuracy
    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
      for inputs, labels in test_loader:
        outputs = net(inputs)
        predicted = torch.argmax(softmax(outputs), dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_acc = correct / total
    print('[{:4d}] train acc: {:.2%} test acc: {:.2%}'.format(
        epoch + 1, train_acc, test_acc))


Nets = {
    'DeepConvNet': DeepConvNet,
    'EEGNet': EEGNet,
}

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # network
  parser.add_argument('-n', '--net', default='EEGNet', choices=Nets.keys())
  parser.add_argument(
      '-act', '--activation', default='relu', choices=Activations.keys())
  # EEGNet hyper-parameter
  parser.add_argument('-ks', '--kernel_size', default=51, type=int)
  parser.add_argument('-p', '--drop_prob', default=0.25, type=float)
  parser.add_argument('-F1', '--F1', default=16, type=int)
  parser.add_argument('-D', '--D', default=2, type=int)
  parser.add_argument('-flat_dim', '--flat_dim', default=736, type=int)
  # training
  parser.add_argument('-d', '--device', default='cuda')
  parser.add_argument('-bs', '--batch_size', default=64, type=int)
  parser.add_argument('-e', '--epochs', default=600, type=int)
  parser.add_argument('-lr', '--lr', default=0.001, type=float)
  parser.add_argument('-wd', '--weight_decay', default=0, type=float)

  args = parser.parse_args()
  main(args)
