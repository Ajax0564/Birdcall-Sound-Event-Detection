

import numpy as np


def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))


def time_shift_spectrogram(spectrogram):
    
    
    """ 
    https://github.com/johnmartinsson/bird-species-classification/wiki/Data-Augmentation
    Shift a spectrogram along the time axis in the spectral-domain at random
    """
    nb_cols = spectrogram.shape[1]
    nb_shifts = np.random.randint(0, nb_cols)

    return np.roll(spectrogram, nb_shifts, axis=1)


def pitch_shift_spectrogram(spectrogram):
    """ 
    https://github.com/johnmartinsson/bird-species-classification/wiki/Data-Augmentation
    Shift a spectrogram along the frequency axis in the spectral-domain at random
    """
    nb_cols = spectrogram.shape[0]
    max_shifts = nb_cols//20 # around 5% shift
    nb_shifts = np.random.randint(-max_shifts, max_shifts)

    return np.roll(spectrogram, nb_shifts, axis=0)


def same_class_augmentation(spectrogram, ebrid_code, file_list):
    
    """
    modified from 
    https://github.com/johnmartinsson/bird-species-classification/wiki/Data-Augmentation
    
    """
    
    same_cls_idx = np.where(np.array(file_list)[:, 1] == ebrid_code)[0]
    aug_sig_index = np.random.choice(same_cls_idx, 1, replace=False)[0]
    wav_path, _ = file_list[aug_sig_index]
    image = extract_melspectrogram(wav_path)
    height, width, _ = image.shape
    image = cv2.resize(image, (int(width * config.image_size / height), config.image_size))
    aug_sig= (image / 255.0).astype(np.float32)
    alpha = np.random.rand()
    wave = (1.0-alpha)*spectrogram + alpha*aug_sig
    
    return wave


def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False):
    """
    https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html
    """
    cloned = spec.copy()
    num_mel_channels = cloned.shape[0] #H
    
    for i in range(0, num_masks):        
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): 
            return cloned

        mask_end = random.randrange(f_zero, f_zero + f) 
        if (replace_with_zero): 
            cloned[f_zero:mask_end,:,:] = 0
        else: 
            cloned[f_zero:mask_end,:,:] = cloned.mean()
    
    return cloned


def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    
    cloned = spec.copy() #(224, 547, 3)
    len_spectro = cloned.shape[1] #(547)
    
    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): 
            return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero):
            cloned[:,t_zero:mask_end, :] = 0
        else: 
            cloned[:,t_zero:mask_end, :] = cloned.mean()
            
    return cloned


def spec_augment(spec: np.ndarray, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    spec = spec.copy()
    for i in range(num_mask):
        num_freqs, num_frames = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * num_freqs)
        num_frames_to_mask = int(time_percentage * num_frames)
        
        t0 = int(np.random.uniform(low=0.0, high=num_frames - num_frames_to_mask))
        f0 = int(np.random.uniform(low=0.0, high=num_freqs - num_freqs_to_mask))
        
        spec[:, t0:t0 + num_frames_to_mask] = 0     
        spec[f0:f0 + num_freqs_to_mask, :] = 0 
        
    return spec