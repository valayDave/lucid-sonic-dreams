import sys
import os
import shutil
import pickle 
from tqdm import tqdm
import inspect
import numpy as np
import random
from scipy.stats import truncnorm

import torch
import PIL
from PIL import Image
import skimage.exposure
import librosa
import soundfile
import moviepy.editor as mpy
from moviepy.audio.AudioClip import AudioArrayClip
import pygit2
from importlib import import_module
from tqdm import tqdm

from .helper_functions import * 
from .sample_effects import *


def import_stylegan_torch():
    # Clone Official StyleGAN2-ADA Repository
    if not os.path.exists('stylegan2'):
      #pygit2.clone_repository('https://github.com/NVlabs/stylegan2-ada.git',
      #                        'stylegan2')
        pygit2.clone_repository('https://github.com/NVlabs/stylegan2-ada-pytorch.git',
                              'stylegan2')
    # StyleGan2 imports
    sys.path.append("stylegan2")
    import legacy
    import dnnlib


def import_stylegan_tf():
    print("Cloning tensorflow...")
    if not os.path.exists('stylegan2_tf'):
        pygit2.clone_repository('https://github.com/NVlabs/stylegan2-ada.git',
                          'stylegan2_tf')

    #StyleGAN2 Imports
    sys.path.append("stylegan2_tf")
    import dnnlib as dnnlib
    from dnnlib.tflib.tfutil import convert_images_to_uint8 as convert_images_to_uint8
    init_tf()

def import_clmr():
    if not os.path.exists("clmr"):
        pygit2.clone_repository("https://github.com/spijkervet/clmr", "clmr")
    sys.path.append("clmr")

    
    
def show_styles():
    '''Show names of available (non-custom) styles'''

    all_models = consolidate_models()
    styles = set([model['name'].lower() for model in all_models])
    print(*styles, sep='\n')


class MultiTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list

    def __getitem__(self, i):
        return [t[i] for t in self.tensor_list]

    def __len__(self):
        return len(self.tensor_list[0])


class LucidSonicDream:
  def __init__(self, 
               song: str,
               pulse_audio: str = None,
               motion_audio: str = None,
               class_audio: str = None,
               contrast_audio: str = None,
               flash_audio: str = None,
               style: str = 'wikiart',
               input_shape: int = None,
               num_possible_classes: int = None,
               ): 

      # If style is a function, raise exception if function does not take 
      # noise_batch or class_batch parameters
    if callable(style):
     
        func_sig = list(inspect.getfullargspec(style))[0]

        for arg in ['noise_batch', 'class_batch']:
            if arg not in func_sig:
                sys.exit('func must be a function with parameters '\
                       'noise_batch and class_batch')

        # Raise exception if input_shape or num_possible_classes is not provided
        if (input_shape is None) or (num_possible_classes is None):
            sys.exit('input_shape and num_possible_classes '\
                     'must be provided if style is a function')

    # Define attributes
    self.song = song
    self.song_name = song.split("/")[-1].split(".")[0].replace(".mp3", "").replace(".", "")
    self.pulse_audio = pulse_audio
    self.motion_audio = motion_audio
    self.class_audio = class_audio
    self.contrast_audio = contrast_audio
    self.flash_audio = flash_audio
    self.style = style
    self.input_shape = input_shape or torch.tensor([512])
    self.num_possible_classes = num_possible_classes 
    self.style_exists = False
    
    # some stylegan models cannot be converted to pytorch (wikiart)
    self.use_tf = style in ("wikiart",)
    if self.use_tf:
        #import_stylegan_tf()
        print("Cloning tensorflow...")
        if not os.path.exists('stylegan2_tf'):
            pygit2.clone_repository('https://github.com/NVlabs/stylegan2-ada.git',
                              'stylegan2_tf')

        #StyleGAN2 Imports
        sys.path.append("stylegan2_tf")
        self.dnnlib = import_module("dnnlib")
        #import dnnlib as dnnlib
        #from dnnlib.tflib.tfutil import convert_images_to_uint8
        tflib = import_module("dnnlib.tflib.tfutil")
        self.convert_images_to_uint8 = tflib.convert_images_to_uint8#import_module("dnnlib.tflib.tfutil", fromlist=["convert_images_to_uint8"])
        self.init_tf = tflib.init_tf #import_module("dnnlib.tflib.tfutil", fromlist=["init_tf"])
        self.init_tf()
        #init_tf()
    else:
        #import_stylegan_torch()
        # Clone Official StyleGAN2-ADA Repository
        if not os.path.exists('stylegan2'):
          #pygit2.clone_repository('https://github.com/NVlabs/stylegan2-ada.git',
          #                        'stylegan2')
            pygit2.clone_repository('https://github.com/NVlabs/stylegan2-ada-pytorch.git',
                                  'stylegan2')
        # StyleGan2 imports
        sys.path.append("stylegan2")
        #import legacy
        #import dnnlib
        self.dnnlib = import_module("dnnlib")
        self.legacy = import_module("legacy")
        
    # prep clmr
    import_clmr()
    

  def stylegan_init(self):
    '''Initialize StyleGAN(2) weights'''

    style = self.style

    # Initialize TensorFlow
    #if self.use_tf:
    #    init_tf() 

    # If style is not a .pkl file path, download weights from corresponding URL
    if '.pkl' not in style:
      all_models = consolidate_models()
      all_styles = [model['name'].lower() for model in all_models]

      # Raise exception if style is not valid
      if style not in all_styles:  
        sys.exit('Style not valid. Call show_styles() to see all ' \
        'valid styles, or upload your own .pkl file.')

      download_url = [model for model in all_models \
                      if model['name'].lower() == style][0]\
                      ['download_url']
      weights_file = style + '.pkl'

      # If style .pkl already exists in working directory, skip download
      if not os.path.exists(weights_file):
        print('Downloading {} weights (This may take a while)...'.format(style))
        try:
          download_weights(download_url, weights_file)
        except Exception:
          exc_msg = 'Download failed. Try to download weights directly at {} '\
                    'and pass the file path to the style parameter'\
                    .format(download_url)
          sys.exit(exc_msg)
        print('Download complete')

    else:
      weights_file = style

    # load generator
    if self.use_tf:
        # Load weights
        with open(weights_file, 'rb') as f:
            self.Gs = pickle.load(f)[2]
    else:
        print(f'Loading networks from {weights_file}...')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with self.dnnlib.util.open_url(weights_file) as f:
            self.Gs = self.legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    
    # Auto assign num_possible_classes attribute
    try:
      print(self.Gs.mapping.input_templates)
      self.num_possible_classes = self.Gs.mapping.input_templates[1].shape[1]
    except ValueError:
      print(self.Gs.mapping.static_kwargs.label_size)
      self.num_possible_classes = self.Gs.components.mapping\
                                  .static_kwargs.label_size
    except Exception:
      self.num_possible_classes = 0    
    
  def clmr_init(self):
    # TODO: add start and duration to song_name
    song_name = f"{self.song_name}_{self.start}_{self.duration}_{self.fps}"
    song_preds_path = f"clmr_{song_name}_preds.pt"
    
    if os.path.exists(song_preds_path):
        all_preds = torch.load(song_preds_path)
    else:
        # dependencies: torchaudio_augmentations, simclr
        if not os.path.exists("./clmr_magnatagatune_mlp"):
            import subprocess
            subprocess.run(["wget", "-nc", "https://github.com/Spijkervet/CLMR/releases/download/2.1/clmr_magnatagatune_mlp.zip"])
            subprocess.run(["unzip", "-o", "clmr_magnatagatune_mlp.zip"])
            #!wget -nc https://github.com/Spijkervet/CLMR/releases/download/2.1/clmr_magnatagatune_mlp.zip
            #!unzip -o clmr_magnatagatune_mlp.zip

        ENCODER_CHECKPOINT_PATH = "./clmr_magnatagatune_mlp/clmr_epoch=10000.ckpt"
        FINETUNER_CHECKPOINT_PATH = "./clmr_magnatagatune_mlp/mlp_epoch=34-step=2589.ckpt"
        SAMPLE_RATE = 22050

        from clmr.datasets import get_dataset
        from clmr.data import ContrastiveDataset
        from clmr.evaluation import evaluate
        from clmr.models import SampleCNN
        from clmr.modules import ContrastiveLearning, LinearEvaluation, PlotSpectogramCallback
        from clmr.utils import yaml_config_hook, load_encoder_checkpoint, load_finetuner_checkpoint

        import argparse

        device = torch.device("cuda")

        parser = argparse.ArgumentParser(description="CLMR")
        config = yaml_config_hook("./clmr/config/config.yaml")
        for k, v in config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))
        args = parser.parse_args([])
        args.accelerator = None

        n_classes = 50
        encoder = SampleCNN(
            strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
            supervised=args.supervised,
            out_dim=n_classes,
        )

        # get dimensions of last fully-connected layer
        n_features = encoder.fc.in_features

        # load the enoder weights from a CLMR checkpoint
        # set the last fc layer to the Identity function, since we attach the
        # fine-tune head seperately

        state_dict = load_encoder_checkpoint(ENCODER_CHECKPOINT_PATH)
        encoder.load_state_dict(state_dict)
        encoder.fc = torch.nn.Identity() #Identity()

        args.finetuner_mlp = True
        module = LinearEvaluation(
            args,
            encoder,
            hidden_dim=n_features,
            output_dim=n_classes,
        )
        state_dict = load_finetuner_checkpoint(FINETUNER_CHECKPOINT_PATH)
        module.model.load_state_dict(state_dict)


        clmr_model = torch.nn.Sequential(encoder,
                                              module.model,
                                              torch.nn.Sigmoid()
                                             ).to(device)
        # resample audio
        print("SR: ", self.sr)
        audio = self.wav
        if self.sr != SAMPLE_RATE:
            print("New SR: ", SAMPLE_RATE)
            audio = librosa.resample(audio, self.sr, SAMPLE_RATE)
        audio = torch.from_numpy(audio)
        aud_len = args.audio_length

        # dataset that splits into overlapping sets of 2.7s
        class PieceDataset(torch.utils.data.Dataset):
            def __init__(self, audio, aud_len, step):
                self.audio = audio
                self.aud_len = aud_len
                self.step = step
                # pad audio by (aud_len - step) such that at each step/frame we now include the new data points
                pad_size = aud_len - step
                self.padded_audio = torch.cat([torch.zeros(pad_size), audio, torch.zeros(aud_len - pad_size)])

            def __len__(self):
                return int(np.ceil(len(self.audio) / self.step))

            def __getitem__(self, i):
                i *= self.step
                if i + self.aud_len > len(self.padded_audio):
                    raise ValueError("Step and or aud_len do not fit the length of the input wav")
                    # i = len(self.audio) - self.aud_len
                return self.padded_audio[i: i + self.aud_len].unsqueeze(0)

        sr = self.sr
        fps = self.fps

        #frame_duration = int(sr / fps - (sr / fps % 64))
        frame_duration = self.frame_duration
        num_frames = np.ceil(len(audio) / frame_duration)
        print("Frame duration: ", frame_duration)
        print("num frames: ", num_frames)

        # make preds
        #ds = torch.utils.data.TensorDataset(splits)
        ds = PieceDataset(audio, aud_len, step=frame_duration)
        dl = torch.utils.data.DataLoader(ds, batch_size=128, pin_memory=True, num_workers=4, shuffle=False)
        all_preds = []
        for batch in tqdm(dl):
            batch = batch.to(device)
            with torch.no_grad():
                preds = clmr_model(batch)
            all_preds.append(preds)
        all_preds = torch.cat(all_preds, dim=0).cpu()
        print(all_preds.shape)

        torch.save(all_preds, song_preds_path)

        #import seaborn as sns
        #import pandas as pd
        #import matplotlib.pyplot as plt
        #df = pd.DataFrame(all_preds, columns=self.tags_magnatagatune).transpose().astype(float)
        #plt.figure(figsize=(14, 14), dpi=300)
        #sns.heatmap(df, cmap="mako")
        #plt.savefig("heatmap.png")
       
    self.tags_magnatagatune = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird', 'country', 'metal', 'female voice', 'choral']
    torch.save(self.tags_magnatagatune, "clmr_tags.pt")
    
    # Calc repr vector per class (except no_vocal, no_voice)
    # TODO: exclude some vectors or only include the top performing/interesting ones
    
    style = self.style.split("/")[-1].split(".")[0].replace(".pkl", "").replace(".", "")
    latent_path = f"magnatagatune_{style}_latents.pt"
    if os.path.exists(latent_path):
        latents = torch.load(latent_path)
    else:
        sys.path.append("../StyleCLIP_modular")
        from style_clip import Imagine
        imagine = Imagine(
                save_progress=False,
                open_folder=False,
                save_video=False,
                opt_all_layers=1,
                lr_schedule=1,
                noise_opt=0,
                epochs=1,
                iterations=1000,
                batch_size=64,
                style=self.style,
        )
        latents = []
        for tag in tqdm(self.tags_magnatagatune, position=0):
            tqdm.write(tag)
            imagine.set_clip_encoding(text=tag)
            # train
            imagine()
            # save trained results
            w_opt = imagine.model.model.w_opt.detach().cpu()
            latents.append(w_opt)
            # reset
            imagine.reset()
        latents = torch.cat(latents, dim=0)
        torch.save(latents, latent_path)
    print("Latents shape: ", latents.shape)  # should be (50, 18, 512) if opt_all_layers 
    
    # create noise vectors! multiply each pred by the latents and average the result
    # clmr preds shape is (len_song, 50)
    if self.clmr_softmax:
        all_preds = torch.softmax(all_preds / self.clmr_softmax_t, dim=-1)
    else:
        # make normed distr over time
        all_preds = (all_preds - all_preds.mean(dim=0, keepdims=True)) / all_preds.std(dim=0, keepdims=True)
        all_preds = (all_preds.clamp(-1.5, 1.5) + 1.5) / 3
        pass
        # make distr per time_step
        #all_preds /= all_preds.sum(dim=-1, keepdims=True)
        
    
    #smooth_range = self.fps // 2  #self.clmr_smooth_range
    # smooth forward in time
    #smoothed = torch.stack([torch.mean(all_preds[i: i + smooth_range], dim=0) for i in range(len(all_preds - smooth_range))])
    # smooth backwards in time
    #smoothed = torch.stack([torch.mean(all_preds[max(i - smooth_range, 0): i + 1], dim=0) for i in range(len(all_preds))])
    # smooth using ema
    
    spec_norm = self.spec_norm_class
    # TODO: use spectral norm to influence how much change is allowed
    
    ema = self.clmr_ema
    ema_value = all_preds[0]
    smoothed = []
    for i in range(len(all_preds)):
        new_val = all_preds[i] * (1 - ema) * spec_norm[i] + ema_value * ema * (1 - spec_norm[i])
        smoothed.append(new_val)
    smoothed = torch.stack(smoothed)
    
    # TODO: Find out decision thresholds for each category by evaluating classifier on whole magnatagatune dataset!
    # TODO: filter out stuff like guitar/piano, no_vocal etc (or only include some good ones)
    include_list = ["techno", "electronic", "classic", "classical", "slow", "fast", "loud", "soft", "quiet", "drums", "ambient"]
    mask = [tag in include_list for tag in self.tags_magnatagatune]
    #latents = latents[mask]
    #smoothed = smoothed[:, mask]
    
    self.noise = torch.stack([(pred.view(len(latents), 1, 1) * latents).sum(dim=0) for pred in smoothed]).numpy()

    
    self.input_shape = latents[0].unsqueeze(0).shape    


  def extract_lyrics_meaning(self, lyrics_path):
    # lyrics sample:
    """
    1
    00:00:02,633 --> 00:00:05,532
    <font color="#FFFFFF"><i>THREE, TWO, ONE.</i></font>

    2
    00:00:05,599 --> 00:00:08,000
    ♪♪ OBIE TRICE
    REAL NAME NO GIMMICKS... ♪
    """
    sys.path.append("../StyleCLIP_modular")

    
    def time_str_to_seconds(time_str):
        hrs_str = time_str.split(":")[0]
        min_str = time_str.split(":")[1]
        seconds_str = time_str.split(":")[2].split(",")[0]
        ms_str = time_str.split(":")[2].split(",")[1]
        return int(hrs_str) * 360 + int(min_str) * 60 + int(seconds_str) + int(ms_str) / 1000
    
    # read file
    with open(lyrics_path, "r+") as f:
        lines = [l for l in f]
    # format and extract lines
    #print(lines[:10])
    texts = []
    start_times = []
    end_times = []
    while len(lines) > 2:
        if lines[0] == "\n":
            continue
        print("next lines")
        print(lines[0].strip("\n"))
        print(lines[1].strip("\n"))
        print(lines[2].strip("\n"))
        count = 4
        # read times
        start_time_str = lines[1].split(" ")[0]
        end_time_str = lines[1].split(" ")[-1]
        # convert to seconds
        start_time = time_str_to_seconds(start_time_str)
        end_time = time_str_to_seconds(end_time_str)

        # read text
        text = lines[2].strip("♪.\n")
        if len(lines) > 3 and lines[3] != "\n":
            text += " " + lines[3].strip("♪.\n")
            count += 1
            if lines[4] != "\n":
                text += " " + lines[4].strip("♪.\n")
                count += 1
                if lines[5] != "\n":
                    text += " " + lines[5].strip("♪.\n")
                    count += 1
        # remove formatting commands
        while text.find("<") != -1:
            start = text.find("<")
            end = text.find(">")
            text = text[:start] + text[end + 1:]
        # remove spaces and turn to lower-case
        text = text.strip("♪ .\n").lower()

        # save
        texts.append(text)
        start_times.append(start_time)
        end_times.append(end_time)

        # delete lines to move on in file
        del lines[:count]

    #print(texts[:5])
    #print(start_times[:5])
    #print(end_times[:5])
    
    # for each phrase, as add many previous words as long as it still fits into the context length
    phrases = texts
    if self.concat_phrases:
        from style_clip.clip import tokenize

        context_length = 77 #imagine.perceptor.context_length

        phrases = []
        current_phrase = ""
        for phrase in texts:
            current_phrase += " \n " + phrase
            # while the sentence is too long, eliminate the first word. subtract 2 from context length for start and end token
            is_too_long = True
            while is_too_long:
                try:
                    tokens = tokenize(current_phrase, context_length=context_length)
                    is_too_long = False
                except RuntimeError:
                    # tokenizer throws RuntimeError if text is too long
                    current_phrase = " ".join(current_phrase.split(" ")[1:])
            phrases.append(current_phrase)
    
    # calc some stats about the song
    frame_duration = self.frame_duration
    num_frames = np.ceil(len(self.wav) / frame_duration)
    # set path to save/load from
    latent_folder = "lyric_latents"
    os.makedirs(latent_folder, exist_ok=True)
    style = self.style.split("/")[-1].split(".")[0].replace(".pkl", "").replace(".", "")
    iterations = self.lyrics_iterations
    latent_path = f"{latent_folder}/{self.song_name}_lyrics_{style}_latents_it{iterations}{'_concatphrases' if self.concat_phrases else ''}{'' if self.reset_latents_after_phrase else '_noLatReset'}.pt"
    print("Latents at: ", latent_path)
    if os.path.exists(latent_path):
        latents = torch.load(latent_path)
    else:
        from style_clip import Imagine
        imagine = Imagine(
                save_progress=False,
                open_folder=False,
                save_video=False,
                opt_all_layers=1,
                lr_schedule=1,
                noise_opt=0,
                epochs=1,
                iterations=iterations,
                batch_size=32,
                style=self.style,
        )
        
        # calc latents for each phrase
        latents = {"song_start_latent": imagine.model.model.w_opt.detach().cpu()}
        for phrase in tqdm(phrases, position=0):
            tqdm.write(phrase)
            imagine.set_clip_encoding(text=phrase)
            # train
            imagine()
            # save trained results
            w_opt = imagine.model.model.w_opt.detach().cpu()
            latents[phrase] = w_opt
            # reset
            if self.reset_latents_after_phrase:
                imagine.reset()
        # save latents
        torch.save(latents, latent_path)
        
    #song_minutes = librosa.get_duration(self.wav, self.sr)
    #song_seconds = song_minutes * 60

    # smooth spectral norm for smoother transitions
    spec_norm = self.spec_norm_class
    ema_val = 0.75
    ema_spec_norm = []
    val = spec_norm[0]
    for amp in spec_norm:
        val = amp * (1 - ema_val) + ema_val * val
        ema_spec_norm.append(val)
        
    def minmax(a):
        return (a - a.min()) / (a.max() - a.min())
    # calc minmax vals for sigmoid
    temp_vec = torch.arange(0, 10000) / 10000
    temp_vec_sig = torch.sigmoid((temp_vec - 0.5) * self.lyrics_sigmoid_t)
    sig_min, sig_max = temp_vec_sig.min(), temp_vec_sig.max()
    
    # calc mid times
    mid_times = [(end_times[i] * 3 + start_times[i] * 1) / 4 for i in range(len(end_times))]
    
    # assign latents to frames
    start_latent = latents["song_start_latent"]
    current_latent = start_latent
    next_latent = latents[phrases[0]]
    next_mid_time = mid_times[0]
    mid_time_to_mid_time = mid_times[0]
    steps_to_next = int(np.ceil(mid_time_to_mid_time * self.fps))
    current_step = 0
    ampl_sum = sum(ema_spec_norm[: steps_to_next])
    ampl_cumsum = 0
    noise = []
    print("Num frames. ", num_frames)
    print("First num steps to next mid of phrase: ", steps_to_next)
    print("Ampl sum start: ", ampl_sum)
    fracs_before_sig = []
    fracs = []
    for i in range(int(num_frames)):
        current_time = i / self.fps
        #fraction_to_next = (next_mid_time - current_time) / mid_time_to_mid_time
        fraction_to_next = 1 - (current_step / steps_to_next)
        current_step += 1
        
        fracs_before_sig.append(fraction_to_next)
        
        if self.ampl_influences_speed:
            ampl_cumsum += ema_spec_norm[i]
            fraction_to_next = 1 - (ampl_cumsum / ampl_sum)
            
            
        # instead of linear make it a sigmoid such that the space around the text latents is explored for longer
        if self.lyrics_sigmoid_transition:
            # apply sigmoid
            fraction_to_next = torch.sigmoid((torch.tensor(fraction_to_next) - 0.5) * self.lyrics_sigmoid_t).item()
            # norm sigmoid to span between 0 and 1
            fraction_to_next = (fraction_to_next - sig_min) / (sig_max - sig_min)
            fracs.append(fraction_to_next)
            
        interpolated_latent = current_latent * fraction_to_next + next_latent * (1 - fraction_to_next)
        
        noise.append(interpolated_latent.squeeze().numpy())
        
        if len(mid_times) > 1:
            if current_time > mid_times[0]:
                current_text = phrases[0]
                current_latent = latents[current_text]
                
                next_mid_time = mid_times[1]
                mid_time_to_mid_time = next_mid_time - mid_times[0]
                
                next_latent = latents[phrases[1]]
        
                steps_to_next = int(np.ceil(mid_time_to_mid_time * self.fps))
                current_step = 0
                ampl_sum = sum(ema_spec_norm[i: i + steps_to_next])
                ampl_cumsum = 0
                del mid_times[0]
                del phrases[0]
        elif current_time > mid_times[0]:
            # for last image, just show constant frame
            current_text = phrases[0]
            current_latent = latents[current_text]

            next_mid_time = mid_times[0]
            mid_time_to_mid_time = 1

            next_latent = current_latent
    self.noise = np.array(noise)
    print("Noise shape: ", self.noise.shape)
    self.input_shape = start_latent.shape

    

  def load_specs(self):
    '''Load normalized spectrograms and chromagram'''

    start = self.start
    duration = self.duration
    fps = self.fps
    input_shape = self.input_shape
    pulse_percussive = self.pulse_percussive
    pulse_harmonic = self.pulse_harmonic
    motion_percussive = self.motion_percussive
    motion_harmonic = self.motion_harmonic

    # Load audio signal data
    wav, sr = librosa.load(self.song, offset=start, duration=duration)
    wav_motion = wav_pulse = wav_class = wav
    sr_motion = sr_pulse = sr_class = sr

    # If pulse_percussive != pulse_harmonic
    # or motion_percussive != motion_harmonic,
    # decompose harmonic and percussive signals and assign accordingly
    aud_unassigned = (not self.pulse_audio) or (not self.motion_audio)
    pulse_bools_equal = pulse_percussive == pulse_harmonic
    motion_bools_equal = motion_percussive == motion_harmonic

    if aud_unassigned and not all([pulse_bools_equal, motion_bools_equal]):
       wav_harm, wav_perc = librosa.effects.hpss(wav)
       wav_list = [wav, wav_harm, wav_perc]

       pulse_bools = [pulse_bools_equal, pulse_harmonic, pulse_percussive]
       wav_pulse = wav_list[pulse_bools.index(max(pulse_bools))]

       motion_bools = [motion_bools_equal, motion_harmonic, motion_percussive]
       wav_motion = wav_list[motion_bools.index(max(motion_bools))]

    # Load audio signal data for Pulse, Motion, and Class if provided
    if self.pulse_audio:
      wav_pulse, sr_pulse = librosa.load(self.pulse_audio, offset=start, 
                                         duration=duration)
    if self.motion_audio:
      wav_motion, sr_motion = librosa.load(self.motion_audio, offset=start, 
                                           duration=duration)
    if self.class_audio:
      wav_class, sr_class = librosa.load(self.class_audio, offset=start,
                                         duration=duration)
    
    # Calculate frame duration (i.e. samples per frame)
    frame_duration = int(sr / fps - (sr / fps % 64))

    # Generate normalized spectrograms for Pulse, Motion and Class
    self.spec_norm_pulse = get_spec_norm(wav_pulse, sr_pulse, 
                                         input_shape, frame_duration)
    self.spec_norm_motion = get_spec_norm(wav_motion, sr_motion,
                                          input_shape, frame_duration)
    self.spec_norm_class= get_spec_norm(wav_class,sr_class, 
                                        input_shape, frame_duration)

    # Generate chromagram from Class audio
    chrom_class = librosa.feature.chroma_cqt(y=wav_class, sr=sr,
                                             hop_length=frame_duration)
    # Sort pitches based on "dominance"
    chrom_class_norm = chrom_class/\
                       chrom_class.sum(axis = 0, keepdims = 1)
    chrom_class_sum = np.sum(chrom_class_norm,axis=1)
    pitches_sorted = np.argsort(chrom_class_sum)[::-1]

    # Assign attributes to be used for vector generation
    self.wav, self.sr, self.frame_duration = wav, sr, frame_duration
    self.chrom_class, self.pitches_sorted = chrom_class, pitches_sorted


  def transform_classes(self):
    '''Transform/assign value of classes'''
    print("Num classes of model: ", self.num_possible_classes)
    # If model does not use classes, simply return list of 0's
    if self.num_possible_classes == 0:
      self.classes = [0]*12

    else:

      # If list of classes is not provided, generate a random sample
      if self.classes is None: 
        self.classes = random.sample(range(self.num_possible_classes),
                                     min([self.num_possible_classes,12]))
      
      # If length of list < 12, repeat list until length is 12
      if len(self.classes) < 12:
        self.classes = (self.classes * int(np.ceil(12/len(self.classes))))[:12]

      # If dominant_classes_first is True, sort classes accordingly  
      if self.dominant_classes_first:
        self.classes=[self.classes[i] for i in np.argsort(self.pitches_sorted)]


  def update_motion_signs(self):
    '''Update direction of noise interpolation based on truncation value'''
    m = self.motion_react
    t = self.truncation
    motion_signs = self.motion_signs
    current_noise = self.current_noise

    # For each current value in noise vector, change direction if absolute 
    # value +/- motion_react is larger than 2 * truncation
    update = lambda cn, ms: 1 if cn - m < -2 * t else \
                           -1 if cn + m >= 2 * t else ms
    update_vec = np.vectorize(update)

    return update_vec(current_noise, motion_signs)

  def generate_class_vec(self, frame):
    '''Generate a class vector using chromagram, where each pitch 
       corresponds to a class'''

    classes = self.classes 
    chrom_class = self.chrom_class 
    class_vecs = self.class_vecs 
    num_possible_classes = self.num_possible_classes
    class_complexity = self.class_complexity
    class_pitch_react = self.class_pitch_react * 43 / self.fps

    # For the first class vector, simple use values from 
    # the first point in time where at least one pitch > 0 
    # (controls for silence at the start of a track)
    if len(class_vecs) == 0:
      first_chrom = chrom_class[:,np.min(np.where(chrom_class.sum(axis=0) > 0))]
      update_dict = dict(zip(classes, first_chrom))
      class_vec = np.array([update_dict.get(i) \
                            if update_dict.get(i) is not None \
                            else 0 \
                            for i in range(num_possible_classes)])
    
    # For succeeding vectors, update class values scaled by class_pitch_react
    else:
      update_dict = dict(zip(classes, chrom_class[:,frame]))
      class_vec = class_vecs[frame - 1] +\
                  class_pitch_react * \
                  np.array([update_dict.get(i) \
                            if update_dict.get(i) is not None \
                            else 0 \
                            for i in range(num_possible_classes)])
            
    # Normalize class vector between 0 and 1
    if np.where(class_vec != 0)[0].shape[0] != 0:
      class_vec[class_vec < 0] = np.min(class_vec[class_vec >= 0])
      class_vec = (class_vec - np.min(class_vec))/np.ptp(class_vec)

    # If all values in class vector are equal, add 0.1 to first value
    if (len(class_vec) > 0) and (np.all(class_vec == class_vec[0])):
      class_vec[0] += 0.1

    return class_vec * class_complexity
            

  def is_shuffle_frame(self, frame):
    '''Determines if classes should be shuffled in current frame'''

    class_shuffle_seconds = self.class_shuffle_seconds 
    fps = self.fps 

    # If class_shuffle_seconds is an integer, return True if current timestamp
    # (in seconds) is divisible by this integer
    if type(class_shuffle_seconds) == int:
      if frame != 0 and frame % round(class_shuffle_seconds*fps) == 0:
        return True
      else:
        return False 

    # If class_shuffle_seconds is a list, return True if current timestamp 
    # (in seconds) is in list
    if type(class_shuffle_seconds) == list:
      if frame/fps + self.start in class_shuffle_seconds:
        return True
      else:
        return False


  def generate_vectors(self):
    '''Generates noise and class vectors as inputs for each frame'''
    PULSE_SMOOTH = 0.75
    MOTION_SMOOTH = 0.75
    classes = self.classes
    class_shuffle_seconds = self.class_shuffle_seconds or [0]
    class_shuffle_strength = round(self.class_shuffle_strength * 12)
    fps = self.fps
    class_smooth_frames = self.class_smooth_seconds * fps
    motion_react = self.motion_react * 20 / fps

    # Get number of noise vectors to initialize (based on speed_fpm)
    minutes = librosa.get_duration(self.wav, self.sr)
    num_init_noise = round(minutes / 60 * self.speed_fpm)
    
    # If num_init_noise < 2, simply initialize the same 
    # noise vector for all frames 
    if self.use_clmr:
        noise = self.noise
        self.class_vecs = noise
    elif self.visualize_lyrics:
        noise = self.noise
        self.class_vecs = noise
    else:
        if num_init_noise < 2:
            noise = [self.truncation * truncnorm.rvs(-2, 2, size=(1, self.input_shape)).astype(np.float32)[0]] * \
                     len(self.spec_norm_class)

        # Otherwise, initialize num_init_noise different vectors, and generate
        # linear interpolations between these vectors
        else: 
          # Initialize vectors
          init_noise = [self.truncation * truncnorm.rvs(-2, 2, size=(1, self.input_shape)).astype(np.float32)[0] \
                        for i in range(num_init_noise)]

          # Compute number of steps between each pair of vectors
          steps = int(np.floor(len(self.spec_norm_class)) / len(init_noise) - 1)
          print("Steps between vectors", steps)  
          # Interpolate
          noise = full_frame_interpolation(init_noise, 
                                           steps,
                                           len(self.spec_norm_class))
    if self.no_beat:
        return
    print("noise len: ", len(noise))
    # Initialize lists of Pulse, Motion, and Class vectors
    pulse_noise = []
    motion_noise = []
    self.class_vecs = []

    # Initialize "base" vectors based on Pulse/Motion Reactivity values
    pulse_base = np.ones(self.input_shape) * self.pulse_react  #  np.array([self.pulse_react] * self.input_shape)
    motion_base = np.ones(self.input_shape) * motion_react  # np.array([motion_react] * self.input_shape)
    
    # Randomly initialize "update directions" of noise vectors
    self.motion_signs = np.array([random.choice([1, -1]) for _ in range(self.input_shape.numel())]).reshape(self.input_shape)
    #self.motion_signs = np.array([random.choice([1,-1]) \
    #                              for n in range(self.input_shape)])

    # Randomly initialize factors based on motion_randomness
    rand_factors = np.array([random.choice([1, 1 - self.motion_randomness]) for _ in range(self.input_shape.numel())]).reshape(self.input_shape)
    #rand_factors = np.array([random.choice([1, 1 - self.motion_randomness]) \
    #                         for n in range(self.input_shape)])

    

    for i in range(len(self.spec_norm_class)):
      # UPDATE NOISE # 

      # Re-initialize randomness factors every 4 seconds
      if i % round(fps * 4) == 0:
        rand_factors = np.array([random.choice([1, 1 - self.motion_randomness]) for _ in range(self.input_shape.numel())]).reshape(self.input_shape)
        #rand_factors = np.array([random.choice([1, 1 - self.motion_randomness]) \
        #                     for n in range(self.input_shape)])

      # Generate incremental update vectors for Pulse and Motion
      pulse_noise_add = pulse_base * self.spec_norm_pulse[i]
      #print(motion_base, self.spec_norm_motion[i], self.motion_signs.shape, rand_factors.shape)
      motion_noise_add = motion_base * self.spec_norm_motion[i] * \
                         self.motion_signs * rand_factors

      # Smooth each update vector using a weighted average of
      # itself and the previous vector
      if i > 0:
        pulse_noise_add = pulse_noise[i-1] * PULSE_SMOOTH + \
                          pulse_noise_add * (1 - PULSE_SMOOTH)
        motion_noise_add = motion_noise[i - 1] * MOTION_SMOOTH + \
                           motion_noise_add * (1 - MOTION_SMOOTH)

      # Append Pulse and Motion update vectors to respective lists
      pulse_noise.append(pulse_noise_add)
      motion_noise.append(motion_noise_add)
    
      # Update current noise vector by adding current Pulse vector and 
      # a cumulative sum of Motion vectors
      noise[i] = noise[i] + pulse_noise_add + sum(motion_noise[:i+1])
      self.noise = noise
      self.current_noise = noise[i]

      # Update directions
      self.motion_signs = self.update_motion_signs()

      # UPDATE CLASSES #
      # If current frame is a shuffle frame, shuffle classes accordingly
      if self.is_shuffle_frame(i):
        self.classes = self.classes[class_shuffle_strength:] + \
                       self.classes[:class_shuffle_strength]

      # Generate class update vector and append to list
      class_vec_add = self.generate_class_vec(frame = i)
      self.class_vecs.append(class_vec_add)

    # Smoothen class vectors by obtaining the mean vector per 
    # class_smooth_frames frames, and interpolating between these vectors
    if class_smooth_frames > 1:

      # Obtain mean vectors
      class_frames_interp = [np.mean(self.class_vecs[i:i + class_smooth_frames], 
                                     axis = 0) \
                            for i in range(0, len(self.class_vecs), 
                                           class_smooth_frames)]
      # Interpolate
      self.class_vecs = full_frame_interpolation(class_frames_interp, 
                                            class_smooth_frames, 
                                            len(self.class_vecs))
    
    # convert to numpy array:
    self.noise = np.array(self.noise)
    self.class_vecs = np.array(self.class_vecs)

  def setup_effects(self):
    '''Initializes effects to be applied to each frame'''

    self.custom_effects = self.custom_effects or []
    start = self.start
    duration = self.duration

    # Initialize pre-made Contrast effect 
    if all(var is None for var in [self.contrast_audio, 
                                  self.contrast_strength,
                                  self.contrast_percussive]):
      pass
    else:
      self.contrast_audio = self.contrast_audio or self.song
      self.contrast_strength = self.contrast_strength or 0.5
      self.contrast_percussive = self.contrast_percussive or True

      contrast = EffectsGenerator(audio = self.contrast_audio, 
                                  func = contrast_effect, 
                                  strength = self.contrast_strength, 
                                  percussive = self.contrast_percussive)
      self.custom_effects.append(contrast)

    # Initialize pre-made Flash effect
    if all(var is None for var in [self.flash_audio, 
                                  self.flash_strength,
                                  self.flash_percussive]):
      pass
    else:
      self.flash_audio = self.flash_audio or self.song
      self.flash_strength = self.flash_strength or 0.5
      self.flash_percussive = self.flash_percussive or True
  
      flash = EffectsGenerator(audio = self.flash_audio, 
                                  func = flash_effect, 
                                  strength = self.flash_strength, 
                                  percussive = self.flash_percussive)
      self.custom_effects.append(flash)

    # Initialize Custom effects
    for effect in self.custom_effects:
      effect.audio = effect.audio or self.song
      effect.render_audio(start=start, 
                          duration = duration, 
                          n_mels = self.input_shape, 
                          hop_length = self.frame_duration)

  def generate_frames(self):
    '''Generate GAN output for each frame of video'''

    file_name = self.file_name
    resolution = self.resolution
    batch_size = self.batch_size
    num_frame_batches = int(len(self.noise) / batch_size)
    if self.use_tf:
        Gs_syn_kwargs = {'output_transform': {'func': self.convert_images_to_uint8, 
                                          'nchw_to_nhwc': True},
                    'randomize_noise': False,
                    'minibatch_size': batch_size}
    else:
        Gs_syn_kwargs = {'noise_mode': 'const'} # random, const, None

    # Set-up temporary frame directory
    self.frames_dir = file_name.split('.mp4')[0] + '_frames'
    if os.path.exists(self.frames_dir):
        shutil.rmtree(self.frames_dir)
    os.makedirs(self.frames_dir)

    # create dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MultiTensorDataset([torch.from_numpy(self.noise), torch.from_numpy(self.class_vecs)])
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4)

    final_images = []
    file_names = []

    # Generate frames
    for i, (noise_batch, class_batch) in enumerate(tqdm(dl, position=0, desc="Generating frames")):
        # If style is a custom function, pass batches to the function
        if callable(self.style): 
            image_batch = self.style(noise_batch=noise_batch, 
                                   class_batch=class_batch)
        # Otherwise, generate frames with StyleGAN(2)
        else:
            if self.use_tf:
                noise_batch = noise_batch.numpy()
                class_batch = class_batch.numpy()
                w_batch = self.Gs.components.mapping.run(noise_batch, np.tile(class_batch, (batch_size, 1)))
                image_batch = self.Gs.components.synthesis.run(w_batch, **Gs_syn_kwargs)
                image_batch = np.array(image_batch)
            else:
                noise_batch = noise_batch.to(device)
                with torch.no_grad():
                    if self.use_clmr or self.visualize_lyrics:
                        w_batch = noise_batch
                    else:
                        w_batch = self.Gs.mapping(noise_batch, class_batch.to(device), truncation_psi=self.truncation_psi)
                    image_batch = self.Gs.synthesis(w_batch, **Gs_syn_kwargs)
                image_batch = (image_batch.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0).cpu().numpy()

        # For each image in generated batch: apply effects, resize, and save
        for j, array in enumerate(image_batch): 
            image_index = (i * batch_size) + j

            # Apply efects
            for effect in self.custom_effects:
                array = effect.apply_effect(array=array, index=image_index)

            # Save. Include leading zeros in file name to keep alphabetical order
            max_frame_index = num_frame_batches * batch_size + batch_size
            file_name = str(image_index).zfill(len(str(max_frame_index)))
        
            file_names.append(file_name)
            final_images.append(array)
            
            
        if len(final_images) > self.max_frames_in_mem:
            self.store_imgs(file_names, final_images, resolution)
            file_names = []
            final_images = []
    if len(final_images) > 0:
        self.store_imgs(file_names, final_images, resolution)

  def store_imgs(self, file_names, final_images, resolution):
    for file_name, final_image in tqdm(zip(file_names, final_images), position=1, leave=False, desc="Storing frames", total=len(file_names)):
        try:
            with Image.fromarray(final_image, mode='RGB') as final_image_PIL:
                # If resolution is provided, resize
                if resolution:
                    final_image_PIL = final_image_PIL.resize((resolution, resolution))
                final_image_PIL.save(os.path.join(self.frames_dir, file_name + '.jpg'), subsample=0, quality=95)
        except ValueError as e:
            print("An error when saving occurred: ", e)
        

  def hallucinate(self,
                  file_name: str, 
                  output_audio: str = None,
                  fps: int = 43, 
                  resolution: int = None, 
                  start: float = 0, 
                  duration: float = None, 
                  save_frames: bool = False,
                  batch_size: int = 1,
                  speed_fpm: int = 12,
                  pulse_percussive: bool = True,
                  pulse_harmonic: bool = False,
                  pulse_react: float = 0.5,
                  motion_percussive: bool = False,
                  motion_harmonic: bool = True,
                  motion_react: float = 0.5, 
                  motion_randomness: float = 0.5,
                  truncation: float = 1,
                  classes: list = None,
                  dominant_classes_first: bool = False,
                  class_pitch_react: float = 0.5,
                  class_smooth_seconds: int = 1,
                  class_complexity: float = 1, 
                  class_shuffle_seconds: float = None,
                  class_shuffle_strength: float = 0.5,
                  contrast_strength: float = None, 
                  contrast_percussive: bool = None,
                  flash_strength: float = None,
                  flash_percussive: bool = None,
                  custom_effects: list = None,
                  truncation_psi: float = 1.0,
                  max_frames_in_mem: int = 500,
                  no_beat: int = 0,
                  
                  use_clmr=False,
                  clmr_softmax=False,
                  clmr_softmax_t=1.0,
                  clmr_ema=0.9,
                  
                  visualize_lyrics=0,
                  lyrics_path=None,
                  ampl_influences_speed=0,
                  lyrics_sigmoid_transition=0,
                  lyrics_sigmoid_t=8,  # scaling of sigmoid, should be between 5 and 10, the stronger, the quicker the changes between phrases
                  concat_phrases=0,
                  lyrics_iterations=200,
                  reset_latents_after_phrase=1,
                 ):
    '''Full pipeline of video generation'''

    # Raise exception if speed_fpm > fps * 60
    if speed_fpm > fps * 60:
        sys.exit('speed_fpm must not be greater than fps * 60')
    
    # Raise exception if element of custom_effects is not EffectsGenerator
    if custom_effects:
        if not all(isinstance(effect, EffectsGenerator) for effect in custom_effects):
            sys.exit('Elements of custom_effects must be EffectsGenerator objects')

    # Raise exception of classes is an empty list
    if classes:
        if len(classes) == 0:
            sys.exit('classes must be NoneType or list with length > 0')

    # Raise exception if any of the following parameters are not betwee 0 and 1
    for param in ['motion_randomness', 'truncation','class_shuffle_strength', 
                  'contrast_strength', 'flash_strength']:
        if (locals()[param]) and not (0 <= locals()[param] <= 1):
            sys.exit('{} must be between 0 and 1'.format(param))

    self.file_name = file_name if file_name[-4:] == '.mp4' else file_name + '.mp4'
    self.file_name = self.file_name.split("/")[-1].replace(" ", "_")
    self.resolution = resolution
    self.batch_size = batch_size
    self.speed_fpm = speed_fpm
    self.pulse_react = pulse_react
    self.motion_react = motion_react 
    self.motion_randomness = motion_randomness
    self.truncation = truncation
    self.classes = classes
    self.dominant_classes_first = dominant_classes_first
    self.class_pitch_react = class_pitch_react
    self.class_smooth_seconds = class_smooth_seconds
    self.class_complexity = class_complexity
    self.class_shuffle_seconds = class_shuffle_seconds
    self.class_shuffle_strength = class_shuffle_strength
    self.contrast_strength = contrast_strength
    self.contrast_percussive = contrast_percussive
    self.flash_strength = flash_strength
    self.flash_percussive = flash_percussive
    self.custom_effects = custom_effects 
    self.max_frames_in_mem = max_frames_in_mem
    self.no_beat = no_beat
    # stylegan2 params
    self.truncation_psi = truncation_psi
    # clmr params
    self.use_clmr = use_clmr
    self.clmr_softmax = clmr_softmax
    self.clmr_softmax_t = clmr_softmax_t
    self.clmr_ema = clmr_ema
    # lyrics params
    self.visualize_lyrics = visualize_lyrics
    self.ampl_influences_speed = ampl_influences_speed
    self.lyrics_sigmoid_transition = lyrics_sigmoid_transition
    self.lyrics_sigmoid_t = lyrics_sigmoid_t
    self.concat_phrases = concat_phrases
    self.lyrics_iterations = lyrics_iterations
    self.reset_latents_after_phrase = reset_latents_after_phrase

    # Initialize style
    if not self.style_exists:
        print('Preparing style...')
        if not callable(self.style):
          self.stylegan_init()
        self.style_exists = True

    # If there are changes in any of the following parameters,
    # re-initialize audio
    cond_list = [(not hasattr(self, 'fps')) or (self.fps != fps),
                 (not hasattr(self, 'start')) or (self.start != start),
                 (not hasattr(self, 'duration')) or (self.duration != duration),
                 (not hasattr(self, 'pulse_percussive')) or \
                 (self.pulse_percussive != pulse_percussive),
                 (not hasattr(self, 'pulse_harmonic')) or \
                 (self.pulse_percussive != pulse_harmonic),
                 (not hasattr(self, 'motion_percussive')) or \
                 (self.motion_percussive != motion_percussive),
                 (not hasattr(self, 'motion_harmonic')) or \
                 (self.motion_percussive != motion_harmonic)]

    if any(cond_list):
        self.fps = fps
        self.start = start
        self.duration = duration 
        self.pulse_percussive = pulse_percussive
        self.pulse_harmonic = pulse_harmonic
        self.motion_percussive = motion_percussive
        self.motion_harmonic = motion_harmonic

        print('Preparing audio...')
        self.load_specs()

    if self.use_clmr:
        # Make CLMR preds:
        self.clmr_init()
        
    if visualize_lyrics:
        self.extract_lyrics_meaning(lyrics_path)
        
    # Initialize effects
    print('Loading effects...')
    self.setup_effects()
    
    # Transform/assign value of classes
    self.transform_classes()

    # Generate vectors
    print('\n\nDoing math...\n')
    self.generate_vectors()

    # Generate frames
    print('\n\nHallucinating... \n')
    self.generate_frames()

    # Load output audio
    if output_audio:
        wav_output, sr_output = librosa.load(output_audio, offset=start, duration=duration)
    else:
        wav_output, sr_output = self.wav, self.sr

    # Write temporary audio file
    soundfile.write('tmp.wav',wav_output, sr_output)

    # Generate final video
    audio = mpy.AudioFileClip('tmp.wav', fps=self.sr * 2)
    video = mpy.ImageSequenceClip(self.frames_dir, fps=self.sr / self.frame_duration)
    video = video.set_audio(audio)
    video.write_videofile(self.file_name, audio_codec='aac')

    # Delete temporary audio file
    os.remove('tmp.wav')

    # By default, delete temporary frames directory
    if not save_frames: 
      shutil.rmtree(self.frames_dir)


class EffectsGenerator:
  def __init__(self, 
               func, 
               audio: str = None,
               strength: float = 0.5,
               percussive: bool = True):
    self.audio = audio
    self.func = func 
    self.strength = strength
    self.percussive = percussive

    # Raise exception of func does not take in parameters array, 
    # strength, and amplitude
    func_sig = list(inspect.getfullargspec(func))[0]
    for arg in ['array', 'strength', 'amplitude']:
      if arg not in func_sig:
        sys.exit('func must be a function with parameters '\
                 'array, strength, and amplitude')
    

  def render_audio(self, start, duration, n_mels, hop_length):
    '''Prepare normalized spectrogram of audio to be used for effect'''

    # Load spectrogram
    wav, sr = librosa.load(self.audio, offset=start, duration=duration)

    # If percussive = True, decompose harmonic and percussive signals
    if self.percussive: 
      wav = librosa.effects.hpss(wav)[1]

    # Get normalized spectrogram  
    self.spec = get_spec_norm(wav, sr, n_mels=n_mels, hop_length=hop_length)


  def apply_effect(self, array, index):
    '''Apply effect to image (array)'''

    amplitude = self.spec[index]
    return self.func(array=array, strength = self.strength, amplitude=amplitude)
