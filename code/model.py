# -*- coding: utf-8 -*-
"""
@Author: Peilu
@Location: Germany
@Date: 2022/03
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

def _get_clones(module, N):

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Netv4(nn.Module):
    def __init__(self, config):
        super(Netv4, self).__init__()
        
        self.config = config
        self.device=config.device
        
        self.embedding = nn.Embedding(config.vocab_size,config.hidden_size)
        #self.position = nn.Embedding(100, config.hidden_size)
        self.fc = nn.Linear(2048, config.hidden_size*2)
        layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8, dim_feedforward=2048, dropout=0.3,batch_first=True,)
        self.encoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=config.num_layers)
        
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )
        #self._init_embedding()
    def _init_embedding(self):
        extra_tokens = ['a' + str(i + 1) for i in range(15)]
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
        

        for para in model.parameters():
            para.requires_grad = False
        for line in open(self.config.vocab_file, 'r', encoding='utf-8'):
            line = json.loads(line)
            for word, v in line.items():
                if word not in extra_tokens:
                    bert_index = tokenizer(word)['input_ids'][1]
                    bert_emb = model.bert.embeddings.word_embeddings(torch.tensor([bert_index]))
                    self.embedding.weight.data[v] = bert_emb[:,self.config.hidden_size]

        
    def forward(self, input_ids=None, type_ids=None, img=None, query_text=None, query_type=None, feature=None,
                labels=None, mode=None, query_type_ids=None):
        

        #position_ids = torch.tensor(list(range(0,input_ids.size(1)))).to(self.device)
        input_emb = self.embedding(input_ids)+self.embedding(type_ids)+ self.embedding(query_type_ids.to(self.config.device))#+self.position(position_ids)
        
        feature = feature.unsqueeze(1)
        fe = self.fc(feature).view(-1,2,self.config.hidden_size)#.expand(-1, input_emb.size(1), -1)

        f = torch.ones((input_emb.size(0), 1), device=input_ids.device)  # (batch_size,1)
        mask = (torch.cat((f, f, input_ids), dim=1) == 0)  # (batch_size, 1+1+length)
   
        text_img_emb = torch.cat((input_emb[:,:1,:],fe,input_emb[:,1:,:]), dim=1)

        outputs = self.encoder(src=text_img_emb, src_key_padding_mask=mask)

        pooler_output = self.pooler(outputs[:, 0, :])

        return pooler_output


class Netv5(nn.Module):
    def __init__(self, config):
        super(Netv5, self).__init__()
        self.encoder = Netv4(config)
        self.device=config.device
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size+2048, 2),
        )

    def forward(self, input_ids=None, type_ids=None, img=None, query_text=None, query_type=None, feature=None,
                labels=None, mode=None, query_type_ids=None):
        input_ids=input_ids.to(self.device)
        type_ids=type_ids.to(self.device)
        feature=feature.to(self.device)
        
        pooler_output = self.encoder( input_ids=input_ids, type_ids=type_ids, feature=feature, labels=labels, query_type_ids=query_type_ids)

        text_feature = torch.cat((pooler_output, feature), dim=-1)
        outputs = self.classifier(text_feature)
        loss = None
        if labels is not None:
            labels=labels.to(self.device)  
            loss_fct = nn.CrossEntropyLoss()
            #loss_fct = FocalLoss()
            loss = loss_fct(outputs.view(-1, 2), labels)

        return {"outputs": outputs, 'loss': loss}

class Netv6(nn.Module):
    def __init__(self, config):
        super(Netv6, self).__init__()
        config.hidden_size = 512
        self.config = config
        self.device=config.device
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position = nn.Embedding(100, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8, dim_feedforward=2048,
                                                   dropout=0.3,
                                                   batch_first=True, )
        self.img_layers = _get_clones(encoder_layer, N=4)
        self.text_layers = _get_clones(encoder_layer, N=4)
        # self.encoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=6)

        self.classifier = nn.Sequential(
            # nn.Linear(config.hidden_size+2048 , 1024),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 1024),
            nn.Linear(config.hidden_size, 2)
        )

    def forward(self, input_ids=None, type_ids=None, img=None, query_text=None, query_type=None, feature=None, labels=None):
        input_ids=input_ids.to(self.device)
        type_ids=type_ids.to(self.device)
        feature=feature.to(self.device)

        position_ids = torch.tensor(list(range(0,input_ids.size(1)))).to(self.device)
        input_emb = self.embedding(input_ids)+self.embedding(type_ids)+self.position(position_ids)
        
        n = 2
        bottle = torch.randn(input_ids.size(0), n, self.config.hidden_size).to(self.device)

        text_mask=input_ids==0
        bottle_mask=torch.ones((bottle.size(0), bottle.size(1)), device=self.device)# (batch_size,1)
        
        mask = torch.cat((bottle_mask, text_mask), dim=1)  # (batch_size, 1+1+length)
        
        for i in range(len(self.img_layers)):
            out1 = torch.cat((bottle, input_emb), dim=1)
            out1 = self.text_layers[i](out1, src_key_padding_mask=mask)
            out2 = torch.cat((out1[:, :2, :], feature.view(-1, 4, 512)), dim=1)
            out2 = self.img_layers[i](out2)
            bottle = out2[:, :2, :]

        bottle = torch.mean(bottle, 1, True)
        outputs = self.classifier(bottle)
        loss = None
        if labels is not None:
            labels=labels.to(self.device)  
            loss_fct = nn.CrossEntropyLoss()
            # loss_fct = FocalLoss()
            loss = loss_fct(outputs.view(-1, 2), labels)
        return {"outputs": outputs, 'loss': loss}

from torch.autograd import Variable
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.alpha = torch.Tensor([alpha,1-alpha])
        
        self.size_average = size_average

    def forward(self, input, target):
        
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

     
class Netv7(nn.Module):
    def __init__(self,config):
        super(Netv7, self).__init__()
        self.config=config
        config.hidden_size=512
        self.device = config.device
        self.model = nn.Transformer(d_model=config.hidden_size,batch_first=True,num_encoder_layers=3,num_decoder_layers=6)
        self.embedding = nn.Embedding(config.vocab_size,config.hidden_size)
        self.n=12
        self.feature_reform=nn.Linear(2048,12*config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 2),
        )
    def forward(self, input_ids=None, type_ids=None, img=None, query_text=None, query_type=None, feature=None, labels=None):
        # position_ids = torch.tensor(list(range(0,input_ids.size(1)))).to(self.device)
        input_ids = input_ids.to(self.config.device)
        type_ids = type_ids.to(self.config.device)
        feature = feature.to(self.config.device)

        input_emb = self.embedding(input_ids)+self.embedding(type_ids)
        text_padding_mask = input_ids == 0
        reformed_feature = self.feature_reform(feature).view(feature.size(0), self.n, -1)
        
        out = self.model(input_emb,reformed_feature,src_key_padding_mask=text_padding_mask,memory_key_padding_mask=text_padding_mask)
        outputs = self.classifier(out[:,0])

        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss_fct = nn.CrossEntropyLoss()
            # loss_fct = FocalLoss()
            loss = loss_fct(outputs.view(-1, 2), labels)

        return {"outputs": outputs, 'loss': loss}  
         
class Netv8(nn.Module):
    def __init__(self, config):
        super(Netv8, self).__init__()
        self.config = config
        self.device = config.device
        config.num_class = 2
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # self.position = nn.Embedding(100, config.hidden_size)

        self.n = 4
        self.feature_reform = nn.Linear(2048, config.hidden_size * self.n)

        layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8, dim_feedforward=2048, dropout=0.3,
                                           batch_first=True, )
        self.encoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=config.num_layers)

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )

        self.title_classifier = nn.Sequential(
            nn.Linear(config.hidden_size +len(config.kernel_wins) * config.dim_channel, 2)
        )
        self.attr_classifier = nn.Sequential(
            nn.Linear(config.hidden_size+len(config.kernel_wins) * config.dim_channel, 2)
        )
        self.convs = nn.ModuleList([nn.Conv2d(1, config.dim_channel, (w, config.emb_dim)) for w in config.kernel_wins])
        self.dropout = nn.Dropout(config.dropout_rate)
      

    def forward(self, input_ids=None, type_ids=None, img=None, query_text=None, query_type=None, feature=None,
                labels=None, mode=None, query_type_ids=None):
        input_ids = input_ids.to(self.config.device)
        type_ids = type_ids.to(self.config.device)
        feature = feature.to(self.config.device)

        # position_ids = torch.tensor(list(range(0,input_ids.size(1)))).to(self.device)
        input_emb = self.embedding(input_ids) + self.embedding(query_type_ids.to(self.config.device)) + \
                    self.embedding(type_ids)

        reformed_feature = self.feature_reform(feature).view(feature.size(0), self.n, -1)
        text_img_emb = torch.cat((input_emb[:, :1, :], reformed_feature, input_emb[:, 1:, :]), dim=1)
        
        # cnn 
        emb_x = text_img_emb
        emb_x = emb_x.unsqueeze(1)
        con_x = [conv(emb_x) for conv in self.convs]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x)

        # bert
        text_mask = input_ids == 0
        img_mask = torch.zeros((reformed_feature.size(0), reformed_feature.size(1)), device=reformed_feature.device)
        mask = torch.cat((img_mask, text_mask), dim=1)
        outputs = self.encoder(src=text_img_emb,  src_key_padding_mask=mask )
        pooler_output = self.pooler(outputs[:, 0, :])
    
        text_feature = torch.cat((pooler_output, fc_x, ), dim=-1)

        title_ids = torch.from_numpy(np.array(query_type) == '图文').nonzero().view(-1).to(self.device)
        title_text_feature = torch.index_select(text_feature, dim=0, index=title_ids)
        title_logits = self.title_classifier(title_text_feature).view(-1, 2)

        attr_ids = torch.from_numpy(np.array(query_type) != '图文').nonzero().view(-1).to(self.device)
        attr_text_feature = torch.index_select(text_feature, dim=0, index=attr_ids)
        attr_logits = self.attr_classifier(attr_text_feature).view(-1, 2)

        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            weight = torch.tensor([0.5, 0.5]).to(self.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            # loss_fct = FocalLoss()
            title_loss = loss_fct(title_logits.view(-1, 2), torch.index_select(labels, dim=0, index=title_ids))
            attr_loss = loss_fct(attr_logits.view(-1, 2), torch.index_select(labels, dim=0, index=attr_ids))
            loss = 0.5 * title_loss + 0.5 * attr_loss
        all_logits = torch.zeros((text_feature.size(0), 2)).to(self.device)
        all_logits[title_ids] = title_logits
        all_logits[attr_ids] = attr_logits
        return {"outputs": all_logits, 'loss': loss}

class Netv9(nn.Module):

    def __init__(self, config):
        super(Netv9, self).__init__()
        config.hidden_size = config.emb_dim
        config.num_class = 2
        self.config = config
        # load pretrained embedding in embedding layer.
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        # self.position = nn.Embedding(100, config.hidden_size)

        self.n = 4
        self.feature_reform = nn.Linear(2048, config.hidden_size * self.n)

        # Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, config.dim_channel, (w, config.emb_dim)) for w in config.kernel_wins])
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout_rate)

        # FC layer
        self.fc = nn.Linear(len(config.kernel_wins) * config.dim_channel, config.num_class)

    def forward(self, input_ids=None, type_ids=None, img=None, query_text=None, query_type=None, feature=None,
                labels=None, mode=None, query_type_ids=None):
        input_ids = input_ids.to(self.config.device)
        type_ids = type_ids.to(self.config.device)
        feature = feature.to(self.config.device)

        # position_ids = torch.tensor(list(range(0, input_ids.size(1)))).to(self.config.device)
        input_emb = self.embedding(input_ids)  + self.embedding(query_type_ids.to(self.config.device))  +self.embedding(type_ids) #+ self.position(position_ids)

        reformed_feature = self.feature_reform(feature).view(feature.size(0), self.n, -1)

        f = torch.ones((input_emb.size(0), 1), device=input_ids.device)  # (batch_size,1)
        mask = (torch.cat((f, f, input_ids), dim=1) == 0)  # (batch_size, 1+1+length)

        text_img_emb = torch.cat((input_emb[:, :1, :], reformed_feature, input_emb[:, 1:, :]), dim=1)

        emb_x = text_img_emb
        emb_x = emb_x.unsqueeze(1)

        con_x = [conv(emb_x) for conv in self.convs]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x)
        logits = self.fc(fc_x)
        loss = None
        if labels is not None:
            labels = labels.to(self.config.device)
            weight = torch.tensor([0.5, 0.5]).to(self.config.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            # loss_fct = FocalLoss()
            loss = loss_fct(logits.view(-1, 2), labels)

        return {"outputs": logits, 'loss': loss}
