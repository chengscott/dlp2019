import argparse
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import itertools


class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    nz, ngf = 64, 64
    self.main = nn.Sequential(
        # input is Z, going into a convolution
        nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (nc) x 64 x 64
    )

  def forward(self, x):
    output = self.main(x)
    return output


class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    ndf = 64
    self.main = nn.Sequential(
        # input is (nc) x 64 x 64
        nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 32 x 32
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True))
    self.discriminator = nn.Sequential(
        # state size. (ndf*8) x 4 x 4
        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid())
    self.Q = nn.Sequential(
        nn.Linear(in_features=8192, out_features=100), nn.ReLU(),
        nn.Linear(in_features=100, out_features=10))

  def forward(self, x, Q=False):
    output = self.main(x)
    d = self.discriminator(output).reshape(-1, 1)
    if not Q:
      return d
    output = output.view(-1, 8192)
    q = self.Q(output)
    return d, q


class MNISTLoader(torch.utils.data.Dataset):
  def __init__(self, root, device):
    super().__init__()
    dataset = torchvision.datasets.MNIST(root, download=True)
    # data transforms
    data_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
    # data
    from PIL import Image
    self.data = []
    for data in dataset.data:
      image = Image.fromarray(data.numpy(), mode='L')
      self.data.append(data_transform(image).to(device))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index]


def get_dataloader(batch_size, device):
  loader = MNISTLoader('./MNIST', device)
  return torch.utils.data.DataLoader(
      loader, batch_size=batch_size, shuffle=True)


def train(args, epoch_start=1):
  dataloader = get_dataloader(args.batch_size, args.device)
  c_prob = torch.Tensor([.1] * 10).to(args.device)
  one_hot = torch.eye(10).reshape(10, 10, 1, 1).to(args.device)
  real_labels = torch.full((args.batch_size, 1), 1).to(args.device)
  fake_labels = torch.full((args.batch_size, 1), 0).to(args.device)
  latent_z = torch.randn(8, 54, 1, 1).to(args.device)
  eval_noise = torch.cat(
      (latent_z.repeat_interleave(10, dim=0), one_hot.repeat(8, 1, 1, 1)),
      dim=1)
  batch_loss = []
  # net
  net_G = Generator().to(args.device)
  net_D = Discriminator().to(args.device)
  criterion_Q = nn.CrossEntropyLoss()
  criterion = nn.BCELoss()
  optim_G = torch.optim.Adam(net_G.parameters(), lr=2e-4, betas=(0.5, 0.99))
  optim_D = torch.optim.Adam(net_D.parameters(), lr=1e-3, betas=(0.5, 0.99))
  optim_Q = torch.optim.Adam(
      itertools.chain(net_G.parameters(), net_D.parameters()),
      lr=1e-3,
      betas=(0.5, 0.99))

  for epoch in range(epoch_start, args.epochs + 1):
    for i, data in enumerate(dataloader, start=1):
      batch_size = len(data)
      real_label = real_labels[-batch_size:]
      fake_label = fake_labels[-batch_size:]

      ############################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ############################

      # zero the parameter gradients
      optim_D.zero_grad()

      # train with real
      output = net_D(data)
      loss_D_real = criterion(output, real_label)
      loss_D_real.backward()
      D_x = output.mean().item()

      # random noise
      latent_z = torch.randn(batch_size, 54, 1, 1).to(args.device)
      latent_c = torch.multinomial(c_prob, batch_size, replacement=True)
      one_hot_c = one_hot[latent_c]

      # train with fake
      input_noise = torch.cat((latent_z, one_hot_c), dim=1)
      fake = net_G(input_noise)
      output = net_D(fake.detach())
      loss_D_fake = criterion(output, fake_label)
      loss_D_fake.backward()
      D_G_z1 = output.mean().item()

      # train loss
      loss_D = loss_D_real + loss_D_fake
      optim_D.step()

      ############################
      # (2) Update G network: maximize log(D(G(z)))
      ############################

      # zero the parameter gradients
      optim_G.zero_grad()

      # train G
      output = net_D(fake)
      loss_G = criterion(output, real_label)
      loss_G.backward()
      optim_G.step()
      D_G_z2 = output.mean().item()

      # zero the parameter gradients
      optim_Q.zero_grad()

      # train Q
      _, Q = net_D(net_G(input_noise), Q=True)
      loss_Q = criterion_Q(Q, latent_c)
      loss_Q.backward()
      optim_Q.step()

      batch_loss.append((loss_D.item(), loss_G.item(), loss_Q.item(), D_x,
                         D_G_z1, D_G_z2))
      if i % 100 == 0:
        print(
            '[{}.{}] Loss_D: {:.2f} Loss_G: {:.2f} Loss_Q: {:.2f} D(x): {:.2f} D(G(z)): {:.2f} / {:.2f}'
            .format(epoch, i, *[sum(x) / len(x) for x in zip(*batch_loss)]))
        batch_loss = []

        fake = net_G(eval_noise)
        torchvision.utils.save_image(
            fake.detach(),
            '{}/{:02d}.png'.format(args.output, epoch),
            nrow=10,
            normalize=True)

    # model checkpoint
    torch.save({
        'epoch': epoch,
        'batch': i,
        'netG': net_G.state_dict(),
        'netD': net_D.state_dict(),
        'optimG': optim_G.state_dict(),
        'optimD': optim_D.state_dict(),
    }, f'{args.path}.ckpt.{epoch}')


def test(args):
  assert (args.restore)
  assert (-1 <= args.condition <= 10)
  net_G = Generator().to(args.device)
  net_D = Discriminator().to(args.device)
  # restore
  checkpoint = torch.load(args.path)
  net_G.load_state_dict(checkpoint['netG'])
  net_D.load_state_dict(checkpoint['netD'])
  # noise
  one_hot = torch.eye(10).reshape(10, 10, 1, 1).to(args.device)
  latent_z = torch.randn(args.samples, 54, 1, 1).to(args.device)
  nrow = 10
  if args.condition == -1:
    eval_noise = torch.cat((latent_z.repeat_interleave(10, dim=0),
                            one_hot.repeat(args.samples, 1, 1, 1)),
                           dim=1)
  else:
    nrow = 1
    condition = one_hot[args.condition].reshape(1, 10, 1, 1)
    eval_noise = torch.cat((latent_z, condition.repeat(args.samples, 1, 1, 1)),
                           dim=1)
  # eval
  fake = net_G(eval_noise)
  torchvision.utils.save_image(
      fake.detach(),
      '{}/eval.png'.format(args.output),
      nrow=nrow,
      normalize=True)


def main(args):
  torch.backends.cudnn.benchmark = True
  if args.condition is not None:
    test(args)
  else:
    train(args)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # network
  parser.add_argument('-p', '--path', default='model/model')
  parser.add_argument('-o', '--output', default='output')
  parser.add_argument('-r', '--restore', action='store_true')
  # training
  parser.add_argument('-d', '--device', default='cuda')
  parser.add_argument('-bs', '--batch_size', default=64, type=int)
  parser.add_argument('-e', '--epochs', default=80, type=int)
  # testing
  parser.add_argument('-c', '--condition', type=int)
  parser.add_argument('-s', '--samples', default=8, type=int)

  args = parser.parse_args()
  main(args)