import logging
import os
import sys
import traceback
from PIL import Image

# adapt the repository paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'lama'))
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'lama', 'configs', 'prediction')


from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path=config_path, config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        if sys.platform != 'win32':
            register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device("cpu")

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = os.path.join(
                predict_config.outdir, 
                os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
            )
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])
            if predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = model(batch)                    
                    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)


            # for filename in os.listdir(indir):
            #     file_path = os.path.join(indir, filename)
            #     if os.path.isfile(file_path):
            #         os.remove(file_path)

            

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)

    pil_image = Image.fromarray(cur_res)
            
    # o_dir = "outputs/gradio"
    # o_file = "ldm_removal_pipe.png"
    # pil_image.save(f"{o_dir}/{o_file}")
    # pil_image.save("/usr/prakt/s0073/image-editing/outputs/gradio/eraser_lama_output.png")

def eraser_lama_gradio(input_image, mask_image):
    # Please update your respective paths
    model_path = "/usr/prakt/s0073/image-editing/code/models/big-lama/"
    print(model_path)

    indir = "/usr/prakt/s0073/image-editing/code/inputs/untracked/eraser-lama" 
    if not os.path.exists(indir):
        os.makedirs(indir)

    outdir = "/usr/prakt/s0073/image-editing/code/outputs/untracked/eraser-lama"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    input_image = input_image.convert("RGB")
    filename_image = "lama_gradio.png"
    input_image.save(f"{indir}/{filename_image}")

    mask_image = mask_image.convert("L")
    filename_mask = "lama_gradio_mask001.png"
    mask_image.save(f"{indir}/{filename_mask}")

    command = "python eraser_lama.py model.path={} indir={} outdir={}".format(model_path, indir, outdir)
    os.system(command)

    # pil_image = Image.fromarray(cur_res)

    # pil_image.save("outputs/gradio/eraser_lama_output.png")


    # for filename in os.listdir(indir):
    #     file_path = os.path.join(indir, filename)
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)

def eraser_lama_gradio_pipe(input_image, coord_input_text):
    pass

if __name__ == '__main__':
    main()


# python eraser_lama.py model.path=$(pwd)/models/big-lama indir=$(pwd)/inputs/untracked/eraser-lama outdir=$(pwd)/outputs/untracked/eraser-lama