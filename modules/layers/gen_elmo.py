import os
import torch
import torch.nn as nn
import json
import codecs
from ELMo.src.modules.embedding_layer import EmbeddingLayer
from torch.autograd import Variable
from ELMo.src.modules.elmo import ElmobiLm
from ELMo.src.modules.lstm import LstmbiLm
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

  def forward(self, word_inp, chars_package, mask):
    token_embedding = self.token_embedder(word_inp, chars_package, (mask.size(0), mask.size(1)))
    if self.config['encoder']['name'] == 'elmo':
      mask = Variable(mask).cuda() if self.use_cuda else Variable(mask)
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


# 动态：encoder
# 导入配置文件
with open('ELMo/configs/cnn_50_100_512_4096_sample.json', 'r') as fin:
    config = json.load(fin)
# 导入词表
if config['token_embedder']['char_dim'] > 0:
    char_lexicon = {}
    with codecs.open(os.path.join('ELMo/zhs.model', 'char.dic'), 'r', encoding='utf-8') as fpi:
        for line in fpi:
            tokens = line.strip().split('\t')
            if len(tokens) == 1:
                tokens.insert(0, '\u3000')
            token, i = tokens
            char_lexicon[token] = int(i)
    char_emb_layer = EmbeddingLayer(config['token_embedder']['char_dim'], char_lexicon, fix_emb=False, embs=None)

if config['token_embedder']['word_dim'] > 0:
    word_lexicon = {}
    with codecs.open(os.path.join('ELMo/zhs.model', 'word.dic'), 'r', encoding='utf-8') as fpi:
        for line in fpi:
            tokens = line.strip().split('\t')
            if len(tokens) == 1:
                tokens.insert(0, '\u3000')
            token, i = tokens
            word_lexicon[token] = int(i)
    word_emb_layer = EmbeddingLayer(config['token_embedder']['word_dim'], word_lexicon, fix_emb=False, embs=None)

elmo = Model(config, word_emb_layer, char_emb_layer, use_cuda=True)
