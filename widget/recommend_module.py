from torch import nn as nn
import torch
import numpy as np
from torch.nn.functional import cosine_similarity

class RecurrentEncoder(nn.Module):
    """ A encoder widget with attention mechanism. """

    def __init__(
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super().__init__()
        self.layers = n_layers
        self.layer_stack = SingleEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

    def forward(self, src_seq, tgt_seq, non_pad_mask, attn_mask, return_attns=False): #[text_emb, img_emb, non_pad_mask, attn_mask]
        #img_encoder
        #src_seq = #[batch_size, context_size*max_len, emb_dim]
        #tgt_seq = #[batch_size, context_size*img_len, emb_dim]
        enc_slf_attn_list = []
        enc_output = tgt_seq
        for i in range(self.layers):
            enc_output, enc_slf_attn = self.layer_stack(
                enc_output,
                src_seq,
                non_pad_mask=non_pad_mask,
                attn_mask=attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Encoder(nn.Module):
    """ A encoder widget with attention mechanism. """

    def __init__(
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            SingleEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, tgt_seq, non_pad_mask, attn_mask, return_attns=False): #[text_emb, img_emb, non_pad_mask, attn_mask]
        #img_encoder
        #src_seq = #[batch_size, context_size*max_len, emb_dim]
        #tgt_seq = #[batch_size, context_size*img_len, emb_dim]
        enc_slf_attn_list = []
        enc_output = tgt_seq
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                src_seq,
                non_pad_mask=non_pad_mask,
                attn_mask=attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class SingleEncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(SingleEncoderLayer, self).__init__()
        self.multi_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, query, kv, non_pad_mask=None, attn_mask=None):
        enc_output, enc_slf_attn = self.multi_attn(query, kv, kv, mask=attn_mask)

        if non_pad_mask != None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)

        if non_pad_mask != None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)
    #
    #     self.init_param()
    #
    # def init_param(self):
    #     nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_k)))
    #     nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_k)))
    #     nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_v)))
    #     nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention
    Attention(Q, K, V)=Softmax(\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right)V)
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            # attn = attn.masked_fill(mask, -np.inf)
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class PositionWiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        # self.w_1 = nn.Linear(d_in, d_hid)
        # self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(torch.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class ImageEncoder(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, in_dim, out_dim, n_head):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for _ in range(n_head)])

    def forward(self, x):

        image_features = []
        for enc_layer in self.layer_stack:
            image_features.append(enc_layer(x))

        output = torch.stack(image_features, dim = 1)

        return output

class RecurrentDSI(nn.Module):
    """ A encoder widget with attention mechanism. """

    def __init__(
            self,
            n_hop, d_model):
        super().__init__()
        self.n_hop = n_hop
        self.txt_layer = nn.Linear(d_model, d_model)
        self.img_layer = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, query_emb, state_emb, type):
        if type == 'txt':
            for hop in range(self.n_hop):
                state_aware_distribution = self.softmax(cosine_similarity(query_emb.unsqueeze(1).repeat(1, len(state_emb[0]), 1), state_emb, dim = -1))
                state_emb = self.txt_layer(state_emb)
                state_aware_emb = torch.matmul(state_aware_distribution.unsqueeze(1), state_emb).squeeze(1)
                query_emb = query_emb + state_aware_emb
        else:
            for hop in range(self.n_hop):
                state_aware_distribution = self.softmax(cosine_similarity(query_emb.unsqueeze(1).repeat(1, len(state_emb[0]), 1), state_emb, dim = -1))
                state_emb = self.img_layer(state_emb)
                state_aware_emb = torch.matmul(state_aware_distribution.unsqueeze(1), state_emb).squeeze(1)
                query_emb = query_emb + state_aware_emb

        state_aware_distribution = self.softmax(cosine_similarity(query_emb.unsqueeze(1).repeat(1, len(state_emb[0]), 1), state_emb, dim = -1))

        return query_emb, state_aware_distribution

class DSI(nn.Module):
    """ A encoder widget with attention mechanism. """

    def __init__(
            self,
            n_hop, d_model):
        super().__init__()
        self.txt_layers = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(n_hop)])
        self.img_layers = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(n_hop)])
        self.softmax = nn.Softmax(dim = -1)
        self.hidden_state = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

    def forward(self, query_emb, state_emb, type):
        # hidden_state = self.hidden_state.repeat(len(state_emb), 1, 1)
        # state_emb = torch.cat([hidden_state, state_emb], dim = 1)
        if type == 'txt':
            for txt_layer in self.txt_layers:
                state_aware_distribution = self.softmax(cosine_similarity(query_emb.unsqueeze(1).repeat(1, len(state_emb[0]), 1), state_emb, dim = -1))
                state_emb = txt_layer(state_emb)
                state_aware_emb = torch.matmul(state_aware_distribution.unsqueeze(1), state_emb).squeeze(1)
                query_emb = query_emb + state_aware_emb
        else:
            for img_layer in self.img_layers:
                state_aware_distribution = self.softmax(cosine_similarity(query_emb.unsqueeze(1).repeat(1, len(state_emb[0]), 1), state_emb, dim = -1))
                state_emb = img_layer(state_emb)
                state_aware_emb = torch.matmul(state_aware_distribution.unsqueeze(1), state_emb).squeeze(1)
                query_emb = query_emb + state_aware_emb

        state_aware_distribution = self.softmax(cosine_similarity(query_emb.unsqueeze(1).repeat(1, len(state_emb[0]), 1), state_emb, dim = -1))

        return query_emb, state_aware_distribution

class DSI_beta(nn.Module):
    """ A encoder widget with attention mechanism. """

    def __init__(
            self,
            n_hop, d_model, temperature):
        super().__init__()
        self.txt_layers = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(n_hop)])
        self.img_layers = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(n_hop)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.softmax = nn.Softmax(dim = -1)
        self.leaky_relu = nn.LeakyReLU()
        self.t = temperature

    def forward(self, query_emb, state_emb, type):
        if type == 'txt':
            for txt_layer in self.txt_layers:
                state_aware_distribution = self.softmax(cosine_similarity(query_emb.unsqueeze(1).repeat(1, len(state_emb[0]), 1), state_emb, dim = -1)*self.t)
                state_emb = self.leaky_relu(txt_layer(state_emb))
                state_aware_emb = torch.matmul(state_aware_distribution.unsqueeze(1), state_emb).squeeze(1)
                query_emb = self.layer_norm(query_emb + state_aware_emb)
        else:
            for img_layer in self.img_layers:
                state_aware_distribution = self.softmax(cosine_similarity(query_emb.unsqueeze(1).repeat(1, len(state_emb[0]), 1), state_emb, dim = -1)*self.t)
                state_emb = self.leaky_relu(img_layer(state_emb))
                state_aware_emb = torch.matmul(state_aware_distribution.unsqueeze(1), state_emb).squeeze(1)
                query_emb = self.layer_norm(query_emb + state_aware_emb)

        state_aware_distribution = self.softmax(cosine_similarity(query_emb.unsqueeze(1).repeat(1, len(state_emb[0]), 1), state_emb, dim = -1)*self.t)

        return query_emb, state_aware_distribution