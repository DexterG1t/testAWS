from flask import Flask, request, render_template, redirect, url_for, send_file
from markupsafe import escape

def png2vector(s, count):
    from PIL import Image
    import operator
    from collections import deque
    from io import StringIO

    def add_tuple(a, b):
        return tuple(map(operator.add, a, b))

    def sub_tuple(a, b):
        return tuple(map(operator.sub, a, b))

    def neg_tuple(a):
        return tuple(map(operator.neg, a))

    def direction(edge):
        return sub_tuple(edge[1], edge[0])

    def magnitude(a):
        return int(pow(pow(a[0], 2) + pow(a[1], 2), .5))

    def normalize(a):
        mag = magnitude(a)
        assert mag > 0, "Cannot normalize a zero-length vector"
        return tuple(map(operator.truediv, a, [mag]*len(a)))

    def svg_header(width, height):
        return """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" 
    "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
    <svg width="%d" height="%d"
        xmlns="http://www.w3.org/2000/svg" version="1.1">
    """ % (width, height)    

    def joined_edges(assorted_edges, keep_every_point=False):
        pieces = []
        piece = []
        directions = deque([
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            ])
        while assorted_edges:
            if not piece:
                piece.append(assorted_edges.pop())
            current_direction = normalize(direction(piece[-1]))
            while current_direction != directions[2]:
                directions.rotate()
            for i in range(1, 4):
                next_end = add_tuple(piece[-1][1], directions[i])
                next_edge = (piece[-1][1], next_end)
                if next_edge in assorted_edges:
                    assorted_edges.remove(next_edge)
                    if i == 2 and not keep_every_point:
                        # same direction
                        piece[-1] = (piece[-1][0], next_edge[1])
                    else:
                        piece.append(next_edge)
                    if piece[0][0] == piece[-1][1]:
                        if not keep_every_point and normalize(direction(piece[0])) == normalize(direction(piece[-1])):
                            piece[-1] = (piece[-1][0], piece.pop(0)[1])
                            # same direction
                        pieces.append(piece)
                        piece = []
                    break
            else:
                raise Exception ("Failed to find connecting edge")
        return pieces

    def rgba_image_to_svg_contiguous(im, opaque=None, keep_every_point=False):

        # collect contiguous pixel groups
        
        adjacent = ((1, 0), (0, 1), (-1, 0), (0, -1))
        visited = Image.new("1", im.size, 0)
        
        color_pixel_lists = {}

        width, height = im.size
        for x in range(width):
            for y in range(height):
                here = (x, y)
                if visited.getpixel(here):
                    continue
                rgba = im.getpixel((x, y))
                if opaque and not rgba[3]:
                    continue
                piece = []
                queue = [here]
                visited.putpixel(here, 1)
                while queue:
                    here = queue.pop()
                    for offset in adjacent:
                        neighbour = add_tuple(here, offset)
                        if not (0 <= neighbour[0] < width) or not (0 <= neighbour[1] < height):
                            continue
                        if visited.getpixel(neighbour):
                            continue
                        neighbour_rgba = im.getpixel(neighbour)
                        if neighbour_rgba != rgba:
                            continue
                        queue.append(neighbour)
                        visited.putpixel(neighbour, 1)
                    piece.append(here)

                if not rgba in color_pixel_lists:
                    color_pixel_lists[rgba] = []
                color_pixel_lists[rgba].append(piece)

        del adjacent
        del visited

        # calculate clockwise edges of pixel groups

        edges = {
            (-1, 0):((0, 0), (0, 1)),
            (0, 1):((0, 1), (1, 1)),
            (1, 0):((1, 1), (1, 0)),
            (0, -1):((1, 0), (0, 0)),
            }
                
        color_edge_lists = {}

        for rgba, pieces in color_pixel_lists.items():
            for piece_pixel_list in pieces:
                edge_set = set([])
                for coord in piece_pixel_list:
                    for offset, (start_offset, end_offset) in edges.items():
                        neighbour = add_tuple(coord, offset)
                        start = add_tuple(coord, start_offset)
                        end = add_tuple(coord, end_offset)
                        edge = (start, end)
                        if neighbour in piece_pixel_list:
                            continue
                        edge_set.add(edge)
                if not rgba in color_edge_lists:
                    color_edge_lists[rgba] = []
                color_edge_lists[rgba].append(edge_set)

        del color_pixel_lists
        del edges

        # join edges of pixel groups

        color_joined_pieces = {}

        for color, pieces in color_edge_lists.items():
            color_joined_pieces[color] = []
            for assorted_edges in pieces:
                color_joined_pieces[color].append(joined_edges(assorted_edges, keep_every_point))

        s = StringIO()
        s.write(svg_header(*im.size))

        for color, shapes in color_joined_pieces.items():
            for shape in shapes:
                s.write(""" <path d=" """)
                for sub_shape in shape:
                    here = sub_shape.pop(0)[0]
                    s.write(""" M %d,%d """ % here)
                    for edge in sub_shape:
                        here = edge[0]
                        s.write(""" L %d,%d """ % here)
                    s.write(""" Z """)
                s.write(""" " style="fill:rgb%s; fill-opacity:%.3f; stroke:none;" />\n""" % (color[0:3], float(color[3]) / 255))
                
        s.write("""</svg>\n""")
        return s.getvalue()

    def rgba_image_to_svg_pixels(im, opaque=None):
        s = StringIO()
        s.write(svg_header(*im.size))

        width, height = im.size
        for x in range(width):
            for y in range(height):
                here = (x, y)
                rgba = im.getpixel(here)
                if opaque and not rgba[3]:
                    continue
                s.write("""  <rect x="%d" y="%d" width="1" height="1" style="fill:rgb%s; fill-opacity:%.3f; stroke:none;" />\n""" % (x, y, rgba[0:3], float(rgba[2]) / 255))
        s.write("""</svg>\n""")
        return s.getvalue()

    def main():
        for i in range(s, count):
            png = f'staticFiles/logo{i}.png'
            svg = f'staticFiles/logo{i}.svg'
            image = Image.open(png).convert('RGBA')
            svg_image = rgba_image_to_svg_contiguous(image)
            #svg_image = rgba_image_to_svg_pixels(image)
            with open(svg, "w") as text_file:
                text_file.write(svg_image)

    if __name__ == '__main__':
        main()



def mainDru_Gen(s, count):
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

    if request.method == 'POST':

        LogoObject = request.form.get('LogoObject')
  
        print(LogoObject, 'main- work')
    obj = LogoObject
    prompt = f"{obj} in <low-poly-hd-logos-icons>"

    for k in range(s, count):
        num_samples = 1 
        num_rows = 1 

        all_images = [] 
        for _ in range(num_rows):
            images = pipe(prompt, num_images_per_prompt=num_samples, num_inference_steps=50, guidance_scale=8).images  #50 , 8
            all_images.extend(images)

        grid = image_grid(all_images, num_samples, num_rows)
        grid.save(f'staticFiles/logo{k}.png')



app = Flask(__name__,template_folder='templateFiles',static_folder='staticFiles')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')



@app.route('/redirect_page')
def redirect_page():    
    return render_template('edit.html', AboutLogo = request.form.get('AboutLogo'))



@app.route('/submit_form', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        LogoObject = request.form.get('LogoObject')
        #mainDru_Gen(1, 2)
        #png2vector(1, 2)
        print(LogoObject)

    return redirect(url_for('redirect_page'))

 

@app.route('/regenerate_page')
def regenerate_page():
    return render_template('regenerate.html')

all_of_count_regerate = []
@app.route('/submit_form_count_regenerate', methods=['POST'])
def submit_form_count_regenerate():
    if request.method == 'POST':
        RegenerateCount = int(request.form.get('regenCount'))
        print(RegenerateCount)
        all_of_count_regerate.append(RegenerateCount)
        #mainDru_Gen(2,RegenerateCount + 2)
        #png2vector(2,RegenerateCount + 2)
        print(sum(all_of_count_regerate))

    return redirect(url_for('regenerate_page'))


@app.route('/redirect_to_index')
def redirect_to_index():
    return redirect('/')

app.run(host='0.0.0.0', port=4567)