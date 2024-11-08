from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import gc
import os
import torch
import base64
from io import BytesIO
import torch.nn.functional as F
import torchvision as tv
from PIL import Image, ImageFile
from objectRemoval_engine import SimpleLama
import io
import cv2
import glob
import numpy as np
 
from Global.detection_models import networks
from Global.detection_util.util import *
import warnings
from gfpgan import GFPGANer



app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
CORS(app)
app.app_context().push()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simple_lama = SimpleLama(device=device)
# -------------------------------------------- ------------------ -----------------------------------------
# -------------------------------------------- index ------------------------------------------------------
# -------------------------------------------- ------------------ -----------------------------------------
@app.route('/')
def hello_world():
    return "api , online"


#----------------------------------- -----------------------------------------------------------
#----------------------------------- Restore Old images -----------------------------------------


version_gan = 1.3
upscaler_gan = 2
bg_upsampler = 'realesrgan'
bg_tile = 400
ext = "auto"
suffix = None
weight_gan = 0.5

if bg_upsampler == 'realesrgan':
    if not torch.cuda.is_available():  # CPU
        import warnings
        warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                        'If you really want to use it, please modify the corresponding codes.')
        bg_upsampler = None
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        model_gan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model_gan,
            tile=bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=True)  # need to set False in CPU mode
else:
    bg_upsampler = None
arch = 'clean'
channel_multiplier = 2
model_name = 'GFPGANv1.3'
url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
# determine model paths
model_path = os.path.join('/app/experiments/pretrained_models', model_name + '.pth')
if not os.path.isfile(model_path):
    model_path = os.path.join('gfpgan/weights', model_name + '.pth')
if not os.path.isfile(model_path):
    # download pre-trained models from url
    model_path = url

restorer = GFPGANer(
        model_path=model_path,
        upscale=upscaler_gan ,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

config_gpu = 0
config_input_size = "full_size"
restore_model = networks.UNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        sync_bn=True,
        antialiasing=True,
    )

# Get the base path of the script
base_path = os.path.dirname(__file__)
# Define the checkpoint path relative to the base path
checkpoint_path = os.path.join(base_path, "Global/checkpoints/detection/FT_Epoch_latest.pt")

if not os.path.exists(checkpoint_path):
    print("Model file not found. Downloading from link...")
    link = "https://huggingface.co/databuzzword/bringing-old-photos-back-to-life/resolve/main/Global/checkpoints/detection/FT_Epoch_latest.pt"  # replace with your model link
    torch.hub.download_url_to_file(link, checkpoint_path)

checkpoint = torch.load(checkpoint_path, map_location="cpu")


restore_model.load_state_dict(checkpoint["model_state"])
print("model weights loaded")
if config_gpu >= 0:
    restore_model.to(config_gpu)
else: 
    restore_model.cpu()
restore_model.eval()

def data_transforms(img, full_size, method=Image.BICUBIC):
    if full_size == "full_size":
        ow, oh = img.size
        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), method)

    elif full_size == "scale_256":
        ow, oh = img.size
        pw, ph = ow, oh
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == ph) and (w == pw):
            return img
        return img.resize((w, h), method)

def scale_tensor(img_tensor, default_scale=256):
    _, _, w, h = img_tensor.shape
    if w < h:
        ow = default_scale
        oh = h / w * default_scale
    else:
        oh = default_scale
        ow = w / h * default_scale

    oh = int(round(oh / 16) * 16)
    ow = int(round(ow / 16) * 16)

    return F.interpolate(img_tensor, [ow, oh], mode="bilinear")

def blend_mask(img, mask):

    np_img = np.array(img).astype("float")

    return Image.fromarray((np_img * (1 - mask) + mask * 255.0).astype("uint8")).convert("RGB")

def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))


@app.route('/enhancer', methods=['POST'])
def restore_images_using_gan():
    print("[/Enhancer_GAN] : New data arrive")
    data = request.get_json()
    base64Image= data["image"]
    #input_img = base64toopencv(base64Image)
    base64_data = ""
    #extracting the base64 string part 
    comma_index = base64Image.find(',')
    if comma_index != -1:
       base64_string = base64Image[comma_index + 1:]
       base64_data = base64_string
    else:
       print("Comma not found in the data URI.")  
       base64_data = base64Image
    img =  stringToImage(base64_data)

    scratch_image = img
    w, h = scratch_image.size
    transformed_image_PIL = data_transforms(scratch_image, config_input_size)
    scratch_image = transformed_image_PIL.convert("L")
    scratch_image = tv.transforms.ToTensor()(scratch_image)
    scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)
    scratch_image = torch.unsqueeze(scratch_image, 0)
    _, _, ow, oh = scratch_image.shape
    scratch_image_scale = scale_tensor(scratch_image)

    if config_gpu >= 0:
        scratch_image_scale = scratch_image_scale.to(config_gpu)
    else:
        scratch_image_scale = scratch_image_scale.cpu()
    with torch.no_grad():
        P = torch.sigmoid(restore_model(scratch_image_scale))

    P = P.data.cpu()
    P = F.interpolate(P, [ow, oh], mode="nearest")

    tv.utils.save_image(
        (P >= 0.4).float(),
        os.path.join(
            "",
            "restore_mask.png",
        ),
        nrow=1,
        padding=0,
        normalize=True,
    )
    transformed_image_PIL.save(os.path.join("", "restore_orignal.png"))
    gc.collect()
    torch.cuda.empty_cache()

    msk = Image.open("restore_mask.png").convert('L')
    new_image = Image.open("restore_orignal.png")
    result = simple_lama(new_image, msk)
    temp_filename = 'restore_result.png'
    result.save(temp_filename)
    input_img = cv2.imread(temp_filename, cv2.IMREAD_COLOR)
    cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            paste_back=True,
            weight=weight_gan)
    cv2.imwrite(temp_filename, restored_img)
    new_image = Image.open("restore_result.png")
    bio = io.BytesIO()
    new_image.save(bio, "PNG")
    bio.seek(0)
    im_b64 = base64.b64encode(bio.getvalue()).decode()
    
    return jsonify({"bg_image":im_b64})















#-------------------------- Enhancer ------------------------------------------------------



 


 
if __name__ == '__main__':
    # app.run(debug=True, use_reloader=False)
    app.run(host="0.0.0.0", port=5000, debug=True,use_reloader=False)
