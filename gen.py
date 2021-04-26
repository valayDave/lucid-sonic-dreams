import time
#from main import LucidSonicDream
from lucidsonicdreams.main import LucidSonicDream
#from lucidsonicdreams import LucidSonicDream

styles = ["../stylegan2-ada-pytorch/jan_regnart_640.pkl", "../stylegan2-ada-pytorch/madziowa_p_800.pkl", "microscope images", "imagenet", "faces (ffhq config-f)", "../stylegan2-ada-pytorch/VisionaryArt.pkl", "../stylegan2-ada-pytorch/astromaniacmag_160.pkl"]

songs = ["songs/henne_song.mp3", "songs/gnossi_1.mp3"]


def hallus(style, song, **kwargs):
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
    style_name = style.split("/")[-1]
    replace_dict = {" ": "_", "-": "_", "(": "", ")": "", ".pkl": "", ":":"_"}
    for key in replace_dict:
        style_name = style_name.replace(key, replace_dict[key])
    
    L = LucidSonicDream(song=song, style=style)
    song_name = song.split(".")[0]
    L.hallucinate(file_name=f"{time_str}_{song_name}_{style_name}.mp4", **kwargs)

    
hallus(styles[-1], songs[1], duration=None, batch_size=4, pulse_react=0.55, motion_react=1.0, fps=60, speed_fpm=1, start=7) 

quit()
    
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=0, clmr_softmax_t=0.1, clmr_ema=0.0, batch_size=4, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10, duration=30) 
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=0, clmr_softmax_t=0.1, clmr_ema=0.05, batch_size=8, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10, duration=30) 

quit()

#hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.1, clmr_ema=0.4, batch_size=4, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10, duration=30) 
# old style to see networks
hallus(styles[-1], songs[0], duration=None, batch_size=4, pulse_react=0.55, motion_react=1.0, fps=60, speed_fpm=1, start=0) 
hallus(styles[-1], songs[1], duration=None, batch_size=4, pulse_react=0.55, motion_react=1.0, fps=60, speed_fpm=1, start=7) 


quit()
    
hallus(styles[1], songs[0], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.1, clmr_ema=0.2, duration=60, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=0) 
    
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.5, clmr_ema=0.2, duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)    
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.2, clmr_ema=0.2, duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)    
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.1, clmr_ema=0.2, duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)    
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.2, duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10) 


quit()

hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=0, clmr_softmax_t=0.05, clmr_ema=0.3, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=0, clmr_softmax_t=0.05, clmr_ema=0.6, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=0, clmr_softmax_t=0.05, clmr_ema=0.9, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
    
    
quit()

#hallus(styles[0], songs[1], use_clmr=1, clmr_softmax=0, duration=20, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=7)
#hallus(styles[0], songs[1], use_clmr=1, clmr_softmax=1, duration=20, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=7)
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.3, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.6, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.9, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)


quit()
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.99, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.999, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
hallus(styles[1], songs[0], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.9999, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=0)
#hallus(styles[0], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.01, duration=20, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=7)

quit()

#hallus(styles[0], songs[1], duration=None, start=7, batch_size=16, truncation_psi=0.9, pulse_react=1.05, motion_react=0.75, fps=60)
#hallus(styles[1], songs[1], duration=None, start=7, batch_size=16, pulse_react=0.9, motion_react=0.85, fps=60)
#hallus(styles[5], songs[1], duration=None, start=7, batch_size=16, truncation_psi=0.9, pulse_react=1.05, motion_react=0.75, fps=60)
hallus(styles[0], songs[1], duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=60, speed_fpm=1)
hallus(styles[5], songs[1], duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=60, speed_fpm=1)

hallus(styles[1], songs[0], duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=60, speed_fpm=1)
#hallus(styles[1], songs[0], duration=None, batch_size=16, pulse_react=0.7, motion_react=0.9, fps=60, speed_fpm=1)
#hallus(styles[1], songs[0], duration=None, batch_size=16, pulse_react=0.7, motion_react=0.9, fps=60, speed_fpm=12)


quit()

#hallus(styles[1], duration=1)
#hallus(styles[0], duration=1)

#hallus(styles[1])

hallus(styles[-1], duration=60, start=210, speed_fpm=3)
hallus(styles[-1], duration=60, start=210, speed_fpm=24)
#hallus(styles[-1], duration=60, start=210, class_complexity=0.5)
#hallus(styles[-1], duration=60, start=210, class_complexity=0.1)
hallus(styles[-1], duration=60, start=210, fps=24)
hallus(styles[-1], duration=60, start=210, batch_size=4)
hallus(styles[-1], duration=60, start=210, batch_size=16)
hallus(styles[-1], pulse_react=1.2, motion_react=0.7)

