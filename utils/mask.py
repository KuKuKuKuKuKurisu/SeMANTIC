import torch

def get_mask(length, target_length, device):
    """Get mask.

    Args:
        length (int): Length.
        target_length: Target length (batch_size, ).

    torch.arange(length, device=device).expand(
        target_length.size(0), length): [0,1,2...length-1]: [batch, length]


    Returns:
        mask: Mask (batch_size, length).

    """
    return torch.arange(length, device=device).expand(
        target_length.size(0), length) < target_length.unsqueeze(1)

def get_non_pad_mask(seq, padding_id):
    assert seq.dim() == 2
    return seq.ne(padding_id).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q, padding_id):
    """ For masking out the padding part of key sequence. """

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(padding_id)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_img_key_pad_mask(seq_k, seq_q, padding_id):
    """ For masking out the padding part of key sequence. """

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(padding_id)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask