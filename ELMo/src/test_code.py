import json
import os
import logging
import codecs
import torch
import torch.nn as nn
from torch.autograd import Variable
from ELMo.src.modules.elmo import ElmobiLm
from ELMo.src.modules.lstm import LstmbiLm
from ELMo.src.modules.embedding_layer import EmbeddingLayer
from ELMo.src.modules.token_embedder import ConvTokenEmbedder, LstmTokenEmbedder




class Model(nn.Module):
  def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda=False):
    super(Model, self).__init__()
    self.use_cuda = use_cuda
    self.config = config

    if config['token_embedder']['name'].lower() == 'cnn':
      self.token_embedder = ConvTokenEmbedder(config, word_emb_layer, char_emb_layer, use_cuda)
    elif config['token_embedder']['name'].lower() == 'lstm':
      self.token_embedder = LstmTokenEmbedder(config, word_emb_layer, char_emb_layer, use_cuda)

    if config['encoder']['name'].lower() == 'elmo':
      self.encoder = ElmobiLm(config, use_cuda)
    elif config['encoder']['name'].lower() == 'lstm':
      self.encoder = LstmbiLm(config, use_cuda)

    self.output_dim = config['encoder']['projection_dim']

  def forward(self, word_inp, chars_package, mask_package):
    token_embedding = self.token_embedder(word_inp, chars_package, (mask_package[0].size(0), mask_package[0].size(1)))
    if self.config['encoder']['name'] == 'elmo':
      mask = Variable(mask_package[0]).cuda() if self.use_cuda else Variable(mask_package[0])
      encoder_output = self.encoder(token_embedding, mask)
      sz = encoder_output.size()
      token_embedding = torch.cat([token_embedding, token_embedding], dim=2).view(1, sz[1], sz[2], sz[3])
      encoder_output = torch.cat([token_embedding, encoder_output], dim=0)
    elif self.config['encoder']['name'] == 'lstm':
      encoder_output = self.encoder(token_embedding)
    return encoder_output

  def load_model(self, path):
    self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl'),
                                                   map_location=lambda storage, loc: storage))
    self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl'),
                                            map_location=lambda storage, loc: storage))



def create_one_batch(x, word2id, char2id, config, w_max_len, oov='<oov>', pad='<pad>', sort=False):
  batch_size = len(x)
  lst = list(range(batch_size))

  if word2id is not None:
    oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
    assert oov_id is not None and pad_id is not None
    batch_w = []
    for x_i in x:
      tmp = []
      for w in x_i:
          tmp.append(word2id.get(w, oov_id))
      tmp = tmp + [pad_id]*(w_max_len+2-len(tmp))
      batch_w.append(tmp)
  else:
    batch_w = None

  if char2id is not None:
    bow_id, eow_id, oov_id, pad_id = char2id.get('<eow>', None), char2id.get('<bow>', None), char2id.get(oov, None), char2id.get(pad, None)

    assert bow_id is not None and eow_id != None and oov_id is not None and pad_id is not None

    if config['token_embedder']['name'].lower() == 'cnn':
      max_chars = config['token_embedder']['max_characters_per_token']
      tmp = []
      for i in lst:
        tmppp = []
        for w in x[i]:
            if len(w) + 2 > max_chars:
                tmppp.append(w[:max_chars-2])
            else:
                tmppp.append(w)
        tmp.append(tmppp)
      x = tmp
      assert max([len(w) for i in lst for w in x[i]]) + 2 <= max_chars
    elif config['token_embedder']['name'].lower() == 'lstm':
      max_chars = max([len(w) for i in lst for w in x[i]]) + 2  # counting the <bow> and <eow>

    batch_c = []
    for x_i in x:
      tmp = []
      for word in x_i:
        tmppp = [bow_id]
        if word == '<bos>' or word == '<eos>':
          tmppp.append(char2id.get(word))
          tmppp.append(eow_id)
        else:
          for ccc in word:
            tmppp.append(char2id.get(ccc, oov_id))
          tmppp.append(eow_id)
        tmppp = tmppp + [pad_id]*(max_chars-len(tmppp))
        tmp.append(tmppp)
      tmp = tmp + [[pad_id]*max_chars]*(w_max_len+2-len(tmp))
      batch_c.append(tmp)
  else:
    batch_c = None

  return batch_w, batch_c







