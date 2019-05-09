import argparse
import itertools
import torch
import torch.nn as nn
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def compute_bleu(output, reference):
  '''compute BLEU-4 score'''
  sf = SmoothingFunction()
  return sentence_bleu([reference],
                       output,
                       weights=(0.25, 0.25, 0.25, 0.25),
                       smoothing_function=sf.method1)


class HiddenEmbed(nn.Module):
  def __init__(self, cond_size, latent_size, hidden_size):
    super().__init__()
    self.embedding = nn.Embedding(4, cond_size)
    self.linear = nn.Linear(latent_size, hidden_size)

  def forward(self, latent, x, linear=True):
    condition = self.embedding(x).view(1, 1, -1)
    hidden = torch.cat((latent, condition), -1)
    if linear:
      hidden = self.linear(hidden)
    return hidden


class EncoderRNN(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.embedding = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size)

  def forward(self, x, hidden):
    embedded = self.embedding(x).view(1, 1, -1)
    output, hidden = self.gru(embedded, hidden)
    return output, hidden


class VAE(nn.Module):
  def __init__(self, hidden_size, latent_size):
    super().__init__()
    self.mu = nn.Linear(hidden_size, latent_size)
    self.logvar = nn.Linear(hidden_size, latent_size)

  def forward(self, x):
    mu = self.mu(x)
    logvar = self.logvar(x)
    latent = mu + torch.randn_like(mu) * torch.exp(logvar / 2)
    KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return latent, KL_loss


class DecoderRNN(nn.Module):
  def __init__(self, hidden_size, output_size, _):
    super().__init__()
    self.embedding = nn.Embedding(output_size, hidden_size)
    self.act = nn.ReLU()
    self.gru = nn.GRU(hidden_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, x, hidden, encoder_output=None):
    output = self.embedding(x).view(1, 1, -1)
    output = self.act(output)
    output, hidden = self.gru(output, hidden)
    output = self.out(output[0])
    return output, hidden


class AttnDecoderRNN(nn.Module):
  def __init__(self, hidden_size, output_size, max_length):
    super().__init__()
    self.embedding = nn.Embedding(output_size, hidden_size)
    self.dropout = nn.Dropout(p=0.1)
    self.attn = nn.Linear(hidden_size * 2, max_length)
    self.softmax = nn.Softmax(dim=1)
    self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
    self.act = nn.ReLU()
    self.gru = nn.GRU(hidden_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, x, hidden, encoder_output):
    output = self.embedding(x).view(1, 1, -1)
    output = self.dropout(output)
    # attention
    attn_weights = self.attn(torch.cat((output[0], hidden[0]), 1))
    attn_weights = self.softmax(attn_weights)
    attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                             encoder_output.unsqueeze(0))
    output = torch.cat((output[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)
    # output
    output = self.act(output)
    output, hidden = self.gru(output, hidden)
    output = self.out(output[0])
    return output, hidden


def get_data(device='cuda'):
  c2i = {k: v for v, k in enumerate('<>_abcdefghijklmnopqrstuvwxyz')}
  i2c = {k: v for v, k in c2i.items()}

  def word2t(word, end=''):
    word += end
    return torch.LongTensor([[c2i[c]] for c in word]).to(device)

  def tense2t(tense):
    return torch.LongTensor([[tense]]).to(device)

  def get_train_data():
    with open('train.txt') as f:
      train_data = [(word2t(w, '>'), tense2t(i)) for line in f
                    for i, w in enumerate(line.split())]
      max_train = max([x.size(0) for x, _ in train_data])
      return train_data, max_train

  def get_test_data():
    with open('test.txt') as f:
      test_tense = [(0, 3), (0, 2), (0, 1), (0, 1), (3, 1), (0, 2), (3, 0),
                    (2, 0), (2, 3), (2, 1)]
      data = [line.split() for line in f]
      test_data = [((word2t(d, '>'), d, tense2t(td)), (l, tense2t(tl)))
                   for (d, l), (td, tl) in zip(data, test_tense)]
      max_test = max([x.size(0) for (x, _, _), _ in test_data])
      return test_data, max_test

  train_data, max_train = get_train_data()
  test_data, max_test = get_test_data()
  max_length = max(max_train, max_test)  # + 1
  return train_data, test_data, max_length, c2i, i2c


def main(args):
  train_data, test_data, max_length, c2i, i2c = get_data(args.device)
  cond_size, latent_size, hidden_size = 8, 40, args.hidden_size
  vocab_size = len(c2i)
  SOS, EOS, UNK = '<', '>', '_'
  SOS_index, EOS_index = c2i[SOS], c2i[EOS]
  epoch_start = 1
  # net
  hidden_embed = HiddenEmbed(cond_size, latent_size,
                             hidden_size).to(args.device)
  encoder = EncoderRNN(vocab_size, hidden_size).to(args.device)
  vae = VAE(hidden_size, latent_size - cond_size).to(args.device)
  decoder = DecoderRNN(hidden_size, vocab_size, max_length).to(args.device)
  criterion = nn.CrossEntropyLoss()
  parameters = itertools.chain(hidden_embed.parameters(), encoder.parameters(),
                               vae.parameters(), decoder.parameters())
  optimizer = torch.optim.SGD(parameters, lr=args.lr)

  # restore model
  if args.restore:
    print('> Restore from', args.path)
    checkpoint = torch.load(args.path)
    hidden_embed.load_state_dict(checkpoint['hidden_embed'])
    encoder.load_state_dict(checkpoint['encoder'])
    vae.load_state_dict(checkpoint['vae'])
    decoder.load_state_dict(checkpoint['decoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch'] + 1
    loss = checkpoint.get('loss', -1)
    KL_loss = checkpoint.get('KL_loss', -1)
    klw = checkpoint.get('klw', -1)
    bleu = checkpoint.get('BLEU', -1)
    print(
        '> Start from [{:2d}] loss: {:.2f} KL_loss: {:.2f} klw: {:.4f} BLEU: {:.2f}'
        .format(epoch_start, loss, KL_loss, klw, bleu))

  def train():
    def KL_anneal(klw, epoch):
      if epoch < 10:
        return 0
      if epoch == 21:
        return 0.001
      if epoch % 10 == 0:
        return klw + 0.0002
      if epoch % 20 == 0:
        return klw - 0.0005
      return klw

    print('> Start training')
    klw = 0
    teacher_forcing_ratio = args.teacher_forcing_ratio
    for epoch in range(epoch_start, args.epochs + 1):
      total_loss, total_kld = [], []
      klw = KL_anneal(klw, epoch)
      #if epoch > 40:
      #  teacher_forcing_ratio = 0.25

      np.random.shuffle(train_data)
      for data_tensor, tense in train_data:
        data_length = data_tensor.size(0)
        loss = 0

        # zero the parameter gradients
        optimizer.zero_grad()

        # hidden condition embedding
        latent = torch.zeros(1, 1, hidden_size - cond_size).to(args.device)
        encoder_hidden = hidden_embed(latent, tense, linear=False)

        # s2s encoder
        encoder_output = torch.zeros(max_length, hidden_size).to(args.device)
        for ei, data in enumerate(data_tensor):
          output, encoder_hidden = encoder(data, encoder_hidden)
          encoder_output[ei] = output[0, 0]

        # vae
        latent, KL_loss = vae(encoder_hidden)
        loss = klw * KL_loss
        assert (KL_loss.abs().item() <= 1e6), KL_loss

        # hidden condition embedding
        decoder_hidden = hidden_embed(latent, tense)

        # s2s decoder
        decoder_input = torch.tensor([[SOS_index]]).to(args.device)
        use_teacher_forcing = np.random.random() < teacher_forcing_ratio
        for di, data in enumerate(data_tensor):
          decoder_output, decoder_hidden = decoder(decoder_input,
                                                   decoder_hidden,
                                                   encoder_output)
          loss += criterion(decoder_output, data)
          if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_input = data
          else:
            # Without teacher forcing: use its own predictions as the next input
            decoder_input = decoder_output.argmax().detach()
            if decoder_input.item() == EOS_index:
              break

        # optimize
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item() / data_length)
        total_kld.append(KL_loss.item() / data_length)

      # train result
      loss = np.mean(total_loss)
      KL_loss = np.mean(total_kld)
      bleu = test(verbose=(epoch == args.epochs))
      print('[{:2d}] loss: {:.2f} KL_loss: {:.2f} klw: {:.4f} BLEU: {:.2f}'.
            format(epoch, loss, KL_loss, klw, bleu))

      # checkpoint
      torch.save(
          {
              'epoch': epoch,
              'loss': loss,
              'KL_loss': KL_loss,
              'klw': klw,
              'BLEU': bleu,
              'hidden_embed': hidden_embed.state_dict(),
              'encoder': encoder.state_dict(),
              'vae': vae.state_dict(),
              'decoder': decoder.state_dict(),
              'optimizer': optimizer.state_dict(),
          }, f'{args.path}.{epoch}.ckpt')

  def test(verbose=False):
    if verbose:
      print('> Start testing')
    with torch.no_grad():
      total_bleu = []
      for (data_tensor, data_word, data_tense), (label_word,
                                                 label_tense) in test_data:
        data_length = data_tensor.size(0)

        # hidden condition embedding
        latent = torch.zeros(1, 1, hidden_size - cond_size).to(args.device)
        encoder_hidden = hidden_embed(latent, data_tense, linear=False)

        # s2s encoder
        encoder_output = torch.zeros(max_length, hidden_size).to(args.device)
        for ei, data in enumerate(data_tensor):
          output, encoder_hidden = encoder(data, encoder_hidden)
          encoder_output[ei] = output[0, 0]

        # vae
        latent, _ = vae(encoder_hidden)

        # hidden condition embedding
        decoder_hidden = hidden_embed(latent, label_tense)

        # s2s decoder
        decoder_input = torch.tensor([[SOS_index]]).to(args.device)
        decoded_word = []
        for _ in range(max_length):
          decoder_output, decoder_hidden = decoder(decoder_input,
                                                   decoder_hidden,
                                                   encoder_output)
          decoder_input = decoder_output.argmax().detach()
          decoded_word.append(i2c[decoder_input.item()])
          if decoded_word[-1] == EOS:
            decoded_word.pop()
            break
        decoded_word = ''.join(decoded_word)

        # test result
        total_bleu.append(compute_bleu(decoded_word, label_word))
        if verbose:
          print('-' * 20)
          print('encode: ', data_word)
          print('truth:  ', label_word)
          print('predict:', decoded_word)

      return np.mean(total_bleu)

  def gaussian_test():
    w2t = {}
    with open('train.txt') as f:
      w2t = {w: i for line in f for i, w in enumerate(line.split())}

    tenses = [torch.LongTensor([[i]]).to(args.device) for i in range(4)]
    while True:
      latent = torch.randn((1, 1, latent_size - cond_size)).to(args.device)
      decoded_words = []
      count = 0
      for tense, tense_tensor in enumerate(tenses):
        decoder_hidden = hidden_embed(latent, tense_tensor)
        decoder_input = torch.tensor([[SOS_index]]).to(args.device)
        decoded_word = []
        for _ in range(max_length):
          decoder_output, decoder_hidden = decoder(decoder_input,
                                                   decoder_hidden, None)
          decoder_input = decoder_output.argmax().detach()
          decoded_word.append(i2c[decoder_input.item()])
          if decoded_word[-1] == EOS:
            decoded_word.pop()
            break
        decoded_word = ''.join(decoded_word)
        if w2t.get(decoded_word, -1) == tense:
          count += 1
        decoded_words.append(decoded_word)
      if count >= 3:
        print(*decoded_words, sep=', ')
        if count == 4:
          break

  if args.gaussian:
    gaussian_test()
  elif epoch_start <= args.epochs:
    train()
  else:
    test(verbose=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # training
  parser.add_argument('-p', '--path', default='model/model')
  parser.add_argument('-r', '--restore', action='store_true')
  parser.add_argument('-d', '--device', default='cuda')
  parser.add_argument('-e', '--epochs', default=100, type=int)
  parser.add_argument('-g', '--gaussian', action='store_true')
  # network
  parser.add_argument('-hs', '--hidden_size', default=512, type=int)
  parser.add_argument('-tfr',
                      '--teacher_forcing_ratio',
                      default=0.5,
                      type=float)
  parser.add_argument('-eir', '--empty_input_ratio', default=0.1, type=float)
  parser.add_argument('-lr', '--lr', default=0.001, type=float)

  args = parser.parse_args()
  main(args)
