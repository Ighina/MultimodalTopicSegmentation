# -*- coding: utf-8 -*-
"""
All code imported from https://librosa.org/doc/main/_modules/librosa/core/pitch.html#yin

the main change is the fact that the yin function is modified to return also the voicing intensity
as defined in https://ieeexplore.ieee.org/abstract/document/8268981

The final modified yin function is then combined with the librosa melspectrogram function to
obtain all the acoustic features described in https://ieeexplore.ieee.org/abstract/document/8268981,
including the pauses, defined as the portions of the audio in which voicing intensity < 0.5
"""
import warnings

import librosa
import numpy as np

warnings.filterwarnings("default")


def get_pause_durations(voicing_intensities, delta=0.5):
    """
    Get the pause durations and the voiced segments (i.e. voicing intensities where there is no pause detection)

    Parameters
    ----------
    voicing_intensities : np.ndarray [shape=(..., n_frames)]
    delta : int [define the threshold under which a frame is considered as a pause]
    Returns
    -------
    pause_durations : np.ndarray [shape=(n_pauses)]
    voiced_segments : np.ndarray [shape=(n_voiced_segments)]
    """
    pauses = []
    voiced_segments = []
    pause = 0
    add = False
    for sample in voicing_intensities:
        if sample < delta:
            pause += 1
            add = True
        else:
            if add:
                pauses.append(pause)
                pause = 0
                add = False
            voiced_segments.append(sample)

    if not pauses:
        if pause > 0:
            pauses.append(pause)
            voiced_segments.append(0)
        else:
            pauses.append(0)
            voiced_segments = voicing_intensities
    return np.array(pauses), np.array(voiced_segments)


def get_acoustic_features(y, sr, previous_f0s=None, mfcc=False):

    stat_fn = [np.nanmean, np.nanstd]

    statistics = []

    if mfcc:

        x = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50)

        delta_x = librosa.feature.delta(x)

        for fn in stat_fn:
            statistics.extend(fn(x, axis=1).tolist())
            statistics.extend(fn(delta_x, axis=1).tolist())

    else:
        f0, _, voicing_intensity = librosa.pyin(y, fmin=70, fmax=500, sr=sr)

        if sum(np.isnan(f0)) == len(f0):
            f0[np.isnan(f0)] = 0

        # f0[np.isnan(f0)]=0

        pauses, voiced_segments = get_pause_durations(voicing_intensity)

        mel_filter = librosa.feature.melspectrogram(y=y, n_mels=40, sr=sr)

        delta_mel = librosa.feature.delta(mel_filter)

        feats = [f0, pauses, voiced_segments, mel_filter, delta_mel]

        for feat in feats:
            for fn in stat_fn:
                try:
                    statistics.extend(fn(feat, axis=1).tolist())
                except:
                    statistics.append(fn(feat, axis=0))
        # statistics = [fn(feat, axis = 0) for feat in feats for fn in stat_fn]

        if previous_f0s is None:
            pitch_jump = 0
        else:
            pitch_jump = np.nanmean(f0[: len(f0) // 5] / np.nanmean(f0)) - np.nanmean(
                previous_f0s[-len(previous_f0s) // 5 :] / np.nanmean(previous_f0s)
            )
            if np.isnan(pitch_jump):
                print("could not compute pitch jump!")
                pitch_jump = 0

        statistics.append(pitch_jump)

    statistics = np.array(statistics)

    if sum(np.isnan(statistics)) > 0:
        print(statistics)
        print(f0)
        raise ValueError

    return statistics
