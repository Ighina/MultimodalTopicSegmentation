# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 21:14:12 2022

@author: User
"""

import argparse
import json
import os
import pickle
import re
import sys

import librosa
import numpy as np
import openl3
import torch
from joblib import Parallel, delayed

from extract_acoustic_features import get_acoustic_features
from librosa import yin
from speechbrain.pretrained import EncoderClassifier
from transformers import Wav2Vec2Model, Wav2Vec2Processor


def create_uniform_segments(lab_times, segment_duration=1, append_labs=False):

    segments = []
    labs = []
    previous_time = 0
    for time in lab_times:
        diff = float(time[1]) - previous_time
        tot_segments = diff / segment_duration
        if append_labs:
            labs.append([0 for x in range(round(tot_segments))])
            try:
                labs[-1][-1] = 1
            except IndexError:
                labs.append(1)
                segments.append((previous_time, float(time[1])))
        else:
            labs.extend([0 for x in range(round(tot_segments))])
            try:
                labs[-1] = 1
            except IndexError:
                labs.append(1)
                segments.append((previous_time, float(time[1])))

        segments.extend(
            [
                (
                    previous_time + segment_duration * i,
                    previous_time + segment_duration * (i + 1),
                )
                for i in range(round(tot_segments))
            ]
        )

        previous_time = float(time[1])

    return segments, labs


def to_sample(sample_rate, time):
    return int(sample_rate * time)


def to_time(sample_rate, samples):
    return samples / sample_rate


def main(args):

    verbose = args.verbose
    all_labs_dictionary = {}
    if not os.path.exists(args.out_directory):
        os.makedirs(os.path.join(args.out_directory))
        existent_files = []
    else:
        if args.openl3 or args.wav2vec:
            existent_files = os.listdir(os.path.join(args.out_directory))
            # print(existent_files)
        else:
            existent_files = os.listdir(args.out_directory)

        if args.continue_from_check:
            print(
                "Warning: the directory where to store the results already exists! Embeddings from the same files will be skipped..."
            )
        else:
            print(
                "Warning: the directory where to store the results already exists! Embeddings will be saved there: this might overwrite existent files..."
            )

    if args.ecapa:
        if verbose:
            print("Computing ecapa embeddings, rather than xvectors...")
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="dehdeh/spkrec-ecapa-voxceleb",
        )
    elif args.openl3:

        class encoder:
            def __init__(self):
                input_repr, content_type, embedding_size = "mel256", "music", 512
                self.model = openl3.models.load_audio_embedding_model(
                    input_repr, content_type, embedding_size
                )

            def encode_batch(self, audio):
                embs, ts = openl3.get_audio_embedding(
                    audio, 16000, verbose=False, model=self.model
                )
                return embs

        model = encoder()

    elif args.prosodic_feats:
        if verbose:
            print(
                "Using mel-filterbank energies and prosodic feature statistics to generate embeddings."
            )

        class encoder:
            def encode_batch(self, audio, previous_pitches=None):
                return get_acoustic_features(audio, 16000, previous_pitches)

        model = encoder()

    elif args.mfcc:
        if verbose:
            print("Using mel-cepstral coefficients statistics to generate embeddings.")

        class encoder:
            def encode_batch(self, audio):
                return get_acoustic_features(audio, 16000, mfcc=True)

        model = encoder()

    elif args.wav2vec:

        class encoder:
            def __init__(self):
                self.preprocessor = Wav2Vec2Processor.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )
                self.model = Wav2Vec2Model.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )

            def encode_batch(self, audio):
                embs = self.preprocessor(
                    audio, sampling_rate=16000, return_tensors="pt"
                ).input_values
                return self.model(embs).last_hidden_state.detach().cpu().numpy()

        model = encoder()

    elif args.CREPE:
        from TorchCrepeModel import get_timestamp_embeddings, load_model

        class encoder:
            def __init__(self):
                self.model = load_model()

            def encode_batch(self, audio):
                embs, ts = get_timestamp_embeddings(audio, self.model)
                embs = embs.squeeze(0)
                return embs.detach().cpu().numpy()

        model = encoder()
    else:
        model = EncoderClassifier.from_hparams(
            source="pretrained_models/",
            hparams_file="hyperparams.yaml",
            savedir="pretrained_models/spkrec-xvect-voxceleb",
        )

    parallel_processes = os.cpu_count() // 2

    data = []
    times = []

    file_paths = []
    audio_paths = []
    filenames = []

    for root, directory, files in os.walk(args.audio_directory):
        for file in files:
            if file.endswith("mp3") or file.endswith("wav"):
                filename = re.findall("(.+)\\.\\w+$", file)[-1]
            else:
                continue
            filenames.append(filename)

            if filename:
                audio_paths.append(os.path.join(args.audio_directory, file))

    if verbose:
        print("Starting audio embedding extraction...")

    for index, audio_path in enumerate(audio_paths):
        if args.continue_from_check and existent_files:
            current_file = os.path.basename(audio_path)[:-4]
            file_exists = re.findall(current_file, " ".join(existent_files))

            if file_exists:

                print(
                    "File {} exists in target directory: skipping this file".format(
                        file_exists[0] + ".npy"
                    )
                )
                continue

        read_audio = True

        if verbose:
            print("Extracting audio embeddings for file {}".format(audio_paths[index]))

        if read_audio:
            audio, sr = librosa.load(audio_paths[index])

        if sr != 16000:
            if verbose:
                print("Resampling audio to 16000 Hz...")
            audio = librosa.resample(audio, sr, 16000)

        audio_length = to_time(16000, len(audio))

        if args.adaptive_uniform_segmentation:
            uniform_interval = audio_length / 100
        else:
            uniform_interval = args.uniform_interval

        start_index = 0 if args.speechbrain or not args.vad else 1
        end_index = start_index + 1
        print(" daje; )")
        prev_pitches = None

        def extract_fn(index2):

            start = to_sample(16000, uniform_interval * index2)

            end = to_sample(16000, uniform_interval * index2 + 1)

            if args.openl3 or args.prosodic_feats or args.mfcc:
                signal = audio[start:end]
            else:
                signal = torch.from_numpy(audio[start:end])

            if args.prosodic_feats:
                if prev_pitches is None:
                    prev_pitches = yin(signal, fmin=70, fmax=500)

                    mean_embedding = model.encode_batch(signal)
                else:

                    mean_embedding = model.encode_batch(signal, prev_pitches)
                    prev_pitches = yin(signal, fmin=70, fmax=500)

            else:
                try:
                    embeddings = model.encode_batch(signal).squeeze().squeeze()
                except RuntimeError:
                    if args.openl3:
                        signal = audio[start : start + (end - start) // 4]
                        embeddings1 = model.encode_batch(signal).squeeze().squeeze()
                        signal2 = audio[
                            start + (end - start) // 4 : start + (end - start) // 2
                        ]
                        embeddings2 = model.encode_batch(signal2).squeeze().squeeze()
                        signal3 = audio[
                            start + (end - start) // 2 : end - (end - start) // 4
                        ]
                        embeddings3 = model.encode_batch(signal3).squeeze().squeeze()
                        signal4 = audio[end - (end - start) // 4 : end]
                        embeddings4 = model.encode_batch(signal3).squeeze().squeeze()
                        embeddings = (
                            embeddings1 + embeddings2 + embeddings3 + embeddings4
                        ) / 4
                    else:

                        signal = torch.from_numpy(
                            audio[start : start + (end - start) // 12]
                        )
                        embeddings1 = model.encode_batch(signal).squeeze().squeeze()
                        signal2 = torch.from_numpy(
                            audio[
                                start + (end - start) // 12 : start + (end - start) // 6
                            ]
                        )
                        embeddings2 = model.encode_batch(signal2).squeeze().squeeze()
                        signal3 = torch.from_numpy(
                            audio[
                                start + (end - start) // 6 : start + (end - start) // 4
                            ]
                        )
                        embeddings3 = model.encode_batch(signal3).squeeze().squeeze()
                        signal4 = torch.from_numpy(
                            audio[
                                start + (end - start) // 4 : start + (end - start) // 3
                            ]
                        )
                        embeddings4 = model.encode_batch(signal4).squeeze().squeeze()
                        signal5 = torch.from_numpy(
                            audio[
                                start
                                + (end - start) // 3 : start
                                + (end - start) // 12 * 5
                            ]
                        )
                        embeddings5 = model.encode_batch(signal5).squeeze().squeeze()
                        signal6 = torch.from_numpy(
                            audio[
                                start
                                + (end - start) // 12 * 5 : end
                                - (end - start) // 2
                            ]
                        )
                        embeddings6 = model.encode_batch(signal6).squeeze().squeeze()
                        signal7 = torch.from_numpy(
                            audio[
                                end - (end - start) // 2 : end - (end - start) // 12 * 5
                            ]
                        )
                        embeddings7 = model.encode_batch(signal7).squeeze().squeeze()
                        signal8 = torch.from_numpy(
                            audio[
                                end - (end - start) // 12 * 5 : end - (end - start) // 3
                            ]
                        )
                        embeddings8 = model.encode_batch(signal8).squeeze().squeeze()
                        signal9 = torch.from_numpy(
                            audio[end - (end - start) // 3 : end - (end - start) // 4]
                        )
                        embeddings9 = model.encode_batch(signal9).squeeze().squeeze()
                        signal10 = torch.from_numpy(
                            audio[end - (end - start) // 4 : end - (end - start) // 6]
                        )
                        embeddings10 = model.encode_batch(signal10).squeeze().squeeze()
                        signal11 = torch.from_numpy(
                            audio[end - (end - start) // 6 : end - (end - start) // 12]
                        )
                        embeddings11 = model.encode_batch(signal11).squeeze().squeeze()
                        signal12 = torch.from_numpy(
                            audio[end - (end - start) // 12 : end]
                        )
                        embeddings12 = model.encode_batch(signal12).squeeze().squeeze()
                        embeddings = (
                            embeddings1
                            + embeddings2
                            + embeddings3
                            + embeddings4
                            + embeddings5
                            + embeddings6
                            + embeddings7
                            + embeddings8
                            + embeddings9
                            + embeddings10
                            + embeddings11
                            + embeddings12
                        ) / 12

                # if args.ecapa:
                #     assert embeddings.shape[0] == 192, "NOOOOO"

                # else:
                #     if args.openl3:
                #         if len(embeddings.shape) == 1:
                #             embeddings = embeddings.reshape(1, 512)
                #         assert embeddings.shape[1] == 512, "NOOOOO"
                #     elif args.wav2vec:
                #         if len(embeddings.shape) == 1:
                #             embeddings = embeddings.reshape(1, 768)
                #         assert embeddings.shape[1] == 768, "NOOOOO"
                #     elif args.CREPE:
                #         if len(embeddings.shape) == 1:
                #             embeddings = embeddings.reshape(1, 256)
                #         assert embeddings.shape[1] == 256, "NOOOOO"
                #     elif args.mfcc:
                #         if len(embeddings.shape) == 1:
                #             embeddings = embeddings.reshape(1, 200)
                #         assert embeddings.shape[1] == 200, "NOOOOO"
                #     else:
                #         assert embeddings.shape[0] == 512, "NOOOOO"

            if args.openl3 or args.wav2vec or args.CREPE:

                return embeddings

            elif args.prosodic_feats:
                return mean_embedding

            elif args.mfcc:
                return embeddings

            else:
                mean_embedding = embeddings.detach().numpy()

                return mean_embedding

        # parallel_processes
        audio_embeddings = Parallel(n_jobs=1)(
            delayed(extract_fn)(i) for i in range(int(audio_length // uniform_interval))
        )
        audio_embeddings = np.array(audio_embeddings)

        out_file = os.path.join(args.out_directory, filenames[index])

        if verbose:
            print("Embeddings extracted!\nWriting them to {}".format(out_file))

        if args.openl3 or args.wav2vec or args.CREPE:
            if not os.path.exists(os.path.join(args.out_directory, "_mean")):
                os.mkdir(os.path.join(args.out_directory, "_mean"))
                os.mkdir(os.path.join(args.out_directory, "_max"))
                os.mkdir(os.path.join(args.out_directory, "_no_reduction"))
                os.mkdir(os.path.join(args.out_directory, "_mean_std"))
                os.mkdir(os.path.join(args.out_directory, "_max_std"))
                os.mkdir(os.path.join(args.out_directory, "_last"))
                os.mkdir(os.path.join(args.out_directory, "_delta_gap"))

            with open(
                os.path.join(args.out_directory, "_no_reduction", filenames[index])
                + ".pkl",
                "wb",
            ) as f:
                pickle.dump(audio_embeddings, f)
            np.save(
                os.path.join(args.out_directory, "_mean", filenames[index]),
                np.array([e.mean(axis=0) for e in audio_embeddings]),
            )
            np.save(
                os.path.join(args.out_directory, "_max", filenames[index]),
                np.array([np.max(e, axis=0) for e in audio_embeddings]),
            )
            np.save(
                os.path.join(args.out_directory, "_mean_std", filenames[index]),
                np.array(
                    [
                        np.concatenate((e.mean(axis=0), np.std(e, axis=0)))
                        for e in audio_embeddings
                    ]
                ),
            )
            np.save(
                os.path.join(args.out_directory, "_max_std", filenames[index]),
                np.array(
                    [
                        np.concatenate((np.max(e, axis=0), np.std(e, axis=0)))
                        for e in audio_embeddings
                    ]
                ),
            )
            np.save(
                os.path.join(args.out_directory, "_last", filenames[index]),
                np.array([e[-1] for e in audio_embeddings]),
            )
            delta_gap = []
            for index_e, e in enumerate(audio_embeddings):
                try:
                    delta_gap.append(audio_embeddings[index_e + 1][0] - e[-1])
                except IndexError:
                    delta_gap.append(e[-1])
            np.save(
                os.path.join(args.out_directory, "_delta_gap", filenames[index]),
                np.array(delta_gap),
            )

        else:
            np.save(out_file, audio_embeddings)


if __name__ == "__main__":

    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write("error: %s\n" % message)
            self.print_help()
            sys.exit(2)

    parser = MyParser(
        description="Compute audio embeddings and store them in the specified directory"
    )

    parser.add_argument(
        "--audio_directory",
        "-audio",
        type=str,
        help="directory containing the audio to be segmented",
    )

    parser.add_argument(
        "--out_directory",
        "-od",
        default="results",
        type=str,
        help="the directory where to store the segmented texts",
    )

    parser.add_argument(
        "--ecapa",
        "-e",
        action="store_true",
        help="Compute ecapa embeddings instead of xvectors.",
    )

    parser.add_argument(
        "--verbose",
        "-vb",
        action="store_true",
        help="Whether to print messages during running.",
    )

    parser.add_argument(
        "--vad",
        "-vd",
        action="store_false",
        help="If included use uniform segmentation rather than a VAD engine to obtain the audio segments",
    )

    parser.add_argument(
        "--speechbrain",
        "-sb",
        action="store_true",
        help="If included it uses speechbrain rather than INA for audio segmentation (recommended)",
    )

    parser.add_argument(
        "--uniform_interval",
        "-ui",
        type=float,
        default=1.0,
        help="If using uniform segmentation, this argument specifies the time frame of each segment.",
    )

    parser.add_argument(
        "--openl3", action="store_true", help="use openl3 to extract audio embeddings"
    )

    parser.add_argument(
        "--wav2vec",
        action="store_true",
        help="use wav2vec2 to extract audio embeddings",
    )

    parser.add_argument(
        "--CREPE", action="store_true", help="use CREPE to extract audio embeddings"
    )

    parser.add_argument(
        "--prosodic_feats",
        action="store_true",
        help="use prosodic features as audio embeddings",
    )

    parser.add_argument(
        "--mfcc", action="store_true", help="use mfccs as audio embeddings"
    )

    parser.add_argument(
        "--continue_from_check",
        "-cont",
        action="store_true",
        help="Continue from previous experiment by avoiding extracting embeddings for files already present in the output folder.",
    )

    parser.add_argument(
        "--adaptive_uniform_segmentation",
        "-aus",
        action="store_true",
        help="Using uniform segmentation, where the length of each segment will be a hundredth of the total document length",
    )

    args = parser.parse_args()

    main(args)
