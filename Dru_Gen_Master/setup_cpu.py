from huggingface_hub import login

login(token='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')



import os
import torch
import PIL
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4" 
from huggingface_hub import hf_hub_download
repo_id_embeds = "sd-concepts-library/low-poly-hd-logos-icons"
embeds_url = "" 
placeholder_token_string = "" 
downloaded_embedding_folder = "./downloaded_embedding"
if not os.path.exists(downloaded_embedding_folder):
    os.mkdir(downloaded_embedding_folder)
if(not embeds_url):
    embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin")
    token_path = hf_hub_download(repo_id=repo_id_embeds, filename="token_identifier.txt")
with open(f'{downloaded_embedding_folder}/token_identifier.txt', 'r') as file:
    placeholder_token_string = file.read()
learned_embeds_path = f"{downloaded_embedding_folder}/learned_embeds.bin"
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
)
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]
    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)
    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer)


pipe = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
).to("cpu")

print('all instaled!')