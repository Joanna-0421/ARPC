import numpy as np
from torchvision.transforms.functional import to_tensor
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
import PIL.Image as PImage
import torch
from tools.run_infinity import *


def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    # print(f'im size {im.shape}')
    return im.add(im).add_(-1)

def load_img(img_path, args):
    # h_div_w = 1.0
    # h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]
    with open(img_path, 'rb') as f:
        img: PImage.Image = PImage.open(f)
        img = img.convert('RGB')
        w,h = img.size
        h_div_w = h/w
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]
        tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']
        img_B3HW = transform(img, tgt_h, tgt_w)

    return img_B3HW.unsqueeze(0)

def mask_quant(vae, vae_scale_schedule, raw_features, device):
    with torch.amp.autocast('cuda', enabled = False):
        B = raw_features.shape[0]
        if raw_features.dim() == 4:
            codes_out = raw_features.unsqueeze(2) #[bs, vocabulary_size, 1, h, w]
        else:
            codes_out = raw_features
        cum_var_input = 0 #当前编码了的feature，用于计算residual
        gt_all_bit_indices = []
        x_BLC_wo_prefix = []
        for si, (pt, ph, pw, pm) in enumerate(vae_scale_schedule):
            residual = codes_out - cum_var_input
            if si != len(vae_scale_schedule)-1:
                residual = F.interpolate(residual, size=vae_scale_schedule[si][:3], mode=vae.quantizer.z_interplote_down).contiguous()
            quantized, _, bit_indices, loss = vae.quantizer.lfq(residual, pm) # quantized shape: [B, d_vae, 1, h, w], bit_indices shape: [B,1,h,w,d_vae]
            gt_all_bit_indices.append(bit_indices)  
            cum_var_input = cum_var_input + F.interpolate(quantized, size=vae_scale_schedule[-1][:3], mode=vae.quantizer.z_interplote_up).contiguous()
            if si < len(vae_scale_schedule)-1:
                this_scale_input = F.interpolate(cum_var_input, size=vae_scale_schedule[si+1][:3], mode=vae.quantizer.z_interplote_up).contiguous()
                x_BLC_wo_prefix.append(this_scale_input.reshape(*this_scale_input.shape[:2], -1).permute(0,2,1)) # (B,H/2*W/2,4C) or (B,H*W,C)
        gt_ms_idx_Bl = [item.reshape(B, -1, vae.codebook_dim) for item in gt_all_bit_indices]
        x_BLC_wo_prefix = torch.cat(x_BLC_wo_prefix, 1)
    
    return x_BLC_wo_prefix, gt_ms_idx_Bl

def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')
    print(f'prompt={prompt}')
    captions = [prompt]
    tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)    
    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple