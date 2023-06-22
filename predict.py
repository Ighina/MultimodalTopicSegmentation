# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:53:24 2021

@author: Iacopo
"""

import argparse
import os
import pickle
import sys

import librosa
import numpy as np
import torch
from pytorch_lightning import Trainer
from scipy.io.wavfile import write
from torch.utils.data import DataLoader

from EncoderDataset import AudioPortionDatasetInference
from models.lightning_model import TextSegmenter
from utils.load_datasets_precomputed import load_dataset_for_inference
from extract_embeddings_inference import main as extract_embeddings


class BasePredictor:
    def test(self):
        pass

    def predict(self):
        raise NotImplementedError(
            "This method needs to be implemented by the child classes!"
        )

    def create_embeddings(
        self,
        encoder,
        audio_directory,
        out_directory,
        uniform_interval=1,
        adaptive_uniform=False,
        verbose = False,
        continue_from_check=True
    ):
        class MockNamespace:
            def __init__(
                self,
                encoder,
                audio_directory,
                out_directory,
                uniform_interval,
                adaptive_uniform,
                verbose,
                continue_from_check
            ):
                # We have the option to use a voice activity detector to segment the input audio and extract embeddings, but we don't use it in the current implementation
                self.vad = False
                self.speechbrain = True

                # If the name of the encoder does not match any of the below, default to x-vectors
                self.ecapa = True if encoder.lower().startswith("ecapa") else False
                self.openl3 = True if encoder.lower().startswith("openl3") else False
                self.wav2vec = True if encoder.lower().startswith("wav2vec") else False
                self.CREPE = True if encoder.lower().startswith("crepe") else False
                self.prosodic_feats = (
                    True if encoder.lower().startswith("prosodic") else False
                )
                self.mfcc = True if encoder.lower().startswith("mfcc") else False

                self.audio_directory = audio_directory
                self.out_directory = out_directory
                self.uniform_interval = uniform_interval
                self.adaptive_uniform_segmentation = adaptive_uniform
                self.verbose = verbose
                self.continue_from_check = continue_from_check

        args = MockNamespace(
            encoder,
            audio_directory,
            out_directory,
            uniform_interval,
            adaptive_uniform,
            verbose,
            continue_from_check
        )
        extract_embeddings(args)

    @staticmethod
    def clean_embedding_folder(embedding_folder):
        os.remove(embedding_folder)

    def segment_audio(self, audio_file, segmentation, mock_audio=None, mock_sr=None):

        if mock_audio is not None:
            # debugging
            assert mock_sr is not None, "Provide a mock sample rate to debug..."
            x, sr = mock_audio, mock_sr

        else:
            x, sr = librosa.load(audio_file)

        if sr != self.sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=self.sr)

        audio_segs = []
        prev_time = 0
        counter = 0

        if self.adapt:
            for i in range(len(x) // 100, len(x) + 1, len(x) // 100):
                if segmentation[counter]:
                    audio_segs.append((prev_time, i))
                    prev_time = i
                counter += 1
        else:
            for i in range(
                self.sr * int(self.interval), len(x) + 1, self.sr * int(self.interval)
            ):
                try:
                    if segmentation[counter]:
                        audio_segs.append((prev_time, i))
                        prev_time = i
                    counter += 1
                except IndexError:
                    break

            audio_segs.append((prev_time, len(x)))

        return audio_segs, x


class Predictor(BasePredictor):
    """
    Predictor class for using Audio Segmenter with neural audio embeddings.
    The Audio Segmenter is typically a neural networks composed of n stacked BiLSTM layers.
    Other architectures like Transformer are also supported.

    parameters
    -----------
    -----------
    hyperparameter_file: str --> path to file produced by the training script in which various hyperparameters of the model are defined.
    best_model_path: str --> path to the saved checkpoint of the model.
    pca_reduce: bool --> True if using PCA to reduce the input feature space (should match training setting).
    pca_value: int --> Number representing the reduced input dimensionality, in case pca_reduce is set to True.
    adaptive_uniform_interval: bool --> True if using the adaptive uniform basic unit extraction technique, where each file is divided into 100 equally spaced chunks and the embeddings are extracted from those chunks.
    uniform_interval: int --> If not using adaptive uniform approach, this parameter specifies the size of each audio chunk into which the original audio file is divided and from which embeddings are extracted (should match training setting).
    original_audio_extension: str --> the file extension of the original audio files.
    threshold: float --> the threshold over which we output a segment boundary. Default to 0.5. Choose smaller or bigger values if you want to bias the segmenter towards outputting more or less boundaries respectively.
    sr: int --> the sample rate used in training. This is typically 16000 and there should not be any need of changing the default value.

    """

    def __init__(
        self,
        hyperparameter_file,
        best_model_path,
        pca_reduce=False,
        pca_value=167,
        adaptive_uniform_interval=False,
        uniform_interval=1,
        original_audio_extension=".mp3",
        threshold=0.5,
        sr=16000,
    ):

        tagset_size = 2

        with open(hyperparameter_file) as f:
            for line in f.readlines():
                if line.startswith("Sentence encoder"):
                    encoder = line.split()[2]
                elif line.startswith("Neural architecture"):
                    architecture = line.split()[2]
                elif line.startswith("Hidden units"):
                    hu = int(line.split()[2])
                elif line.startswith("Number of layers"):
                    nl = int(line.split()[3])

        self.encoder = encoder
        self.architecture = architecture

        if pca_reduce:
            embedding_dim = args.pca_value

        elif self.encoder.startswith("prosodic"):
            embedding_dim = 167

        elif self.encoder.startswith("openl3_std"):
            embedding_dim = 1024

        elif self.encoder.startswith("wav2vec_std"):
            embedding_dim = 1536

        elif (
            self.encoder.startswith("x-vector")
            or self.encoder.startswith("openl3")
            or self.encoder.startswith("crepe_std")
        ):
            embedding_dim = 512

        elif self.encoder.startswith("crepe"):
            embedding_dim = 256

        elif self.encoder.startswith("mfcc"):
            embedding_dim = 200

        elif self.encoder.startswith("ecapa"):
            embedding_dim = 192

        elif self.encoder.startswith("wav2vec"):
            embedding_dim = 768

        else:
            raise ValueError(
                "Encoder not recognised, use one of the three available options (x-vectors, ecapa or wav2vec)"
            )

        if architecture == "SimpleBiLSTM" or architecture == "BiLSTM":
            bidirectional = True
        elif architecture == "LSTM":
            bidirectional = False
        elif architecture == "Transformer":
            raise NotImplementedError()
        elif architecture == "BiLSTM-CRF" or architecture == "Transformer-CRF":
            raise NotImplementedError()

        try:
            self.model = TextSegmenter.load_from_checkpoint(
                best_model_path,
                architecture=architecture,
                tagset_size=tagset_size,
                embedding_dim=embedding_dim,
                hidden_dim=hu,
                bidirectional=bidirectional,
                lr=1e-3,
                num_layers=nl,
                loss_fn="BinaryCrossEntropy",
                dropout_in=0.0,
                dropout_out=0.0,
                threshold=threshold,
            )
        except KeyError:  # the error originates if we used "CrossEntropy" as loss function, as the architecture is slightly different
            self.model = TextSegmenter.load_from_checkpoint(
                best_model_path,
                architecture=architecture,
                tagset_size=tagset_size,
                embedding_dim=embedding_dim,
                hidden_dim=hu,
                bidirectional=bidirectional,
                lr=1e-3,
                num_layers=nl,
                loss_fn="CrossEntropy",
                dropout_in=0.0,
                dropout_out=0.0,
                threshold=0.5,
            )

        self.adapt = False
        if adaptive_uniform_interval:
            self.adapt = True

        self.interval = uniform_interval
        self.ext = original_audio_extension
        self.th = threshold

        self.sr = sr

    def predict(
        self,
        embedding_folder,
        experiment_name,
        write_audio_segments=True,
        audio_directory=None,
        batch_size=1,
        num_gpus=0,
        verbose=False,
        add_overlap=1
    ):

        assert not os.path.exists(
            experiment_name
        ), "The name of this experiment has already be used: please change experiment name or delete all the existent results from {} folder to use this name".format(
            args.experiment_name
        )

        os.makedirs(experiment_name)

        if not torch.cuda.is_available():
            num_gpus = 0

        self.trainer = Trainer(gpus=num_gpus)

        embeddings, file_names = load_dataset_for_inference(args.embedding_folder)
        if verbose:
            print(f"Segmenting the following files:\n{file_names}")
        os.chdir(experiment_name)

        encoder = self.encoder

        test_dataset = AudioPortionDatasetInference(embeddings, encoder=encoder)

        if self.architecture == "SimpleBiLSTM":
            batch_size = 1

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, collate_fn=test_dataset.collater
        )
        if verbose:
            print("Test loader has: {} documents".format(len(test_dataset)))

        results = self.trainer.predict(self.model, test_loader)
        
        if write_audio_segments:
            assert (
                audio_directory is not None
            ), "If segmenting the input audio, then you need to provide the path to where the audio files are"

            os.makedirs("audio_segments")

            for index, file in enumerate(file_names):
                # audio files should be named the same as the relative embedding file, but with a different extension
                audio_file = os.path.join("..", audio_directory, file[:-4] + self.ext)
                assert os.path.exists(
                    audio_file
                ), f"Could not find the audio file associated to the embedding: {file}! Check the paths you provided to the program..."

                if sum(results[index][0])==0:
                    print(f"Warning: no segment identified in the file {file}! No audio segment will be written in the output directory for this file...")
                    continue

                audio_segments, audio = self.segment_audio(
                    audio_file, results[index][0]
                )

                for index_seg, audio_segment in enumerate(audio_segments):

                    if add_overlap:
                        offset = add_overlap*self.sr
                        offset_start, offset_end = (offset, offset) if index_seg else (0, offset) 

                    write(
                        os.path.join(
                            "audio_segments", file[:-4] + str(index_seg) + self.ext
                        ),
                        self.sr,
                        audio[audio_segment[0]-offset_start : audio_segment[1]+offset_end],
                    )
        
        return results


class LogReg_Predictor(BasePredictor):
    def __init__(
        self,
        best_model_path,
        adaptive_uniform_interval=False,
        uniform_interval=1,
        original_audio_extension=".mp3",
        threshold=0.5,
        sr=16000,
    ):

        self.model = pickle.load(open(best_model_path, "rb"))

        self.adapt = False
        if adaptive_uniform_interval:
            self.adapt = True

        self.interval = uniform_interval
        self.ext = original_audio_extension
        self.th = threshold

        self.sr = sr

    def predict(
        self,
        embedding_folder,
        experiment_name,
        audio_directory=None,
        write_audio_segments=True,
        batch_size=1,
        num_gpus=0,
        verbose=False,
    ):

        assert not os.path.exists(
            experiment_name
        ), "The name of this experiment has already be used: please change experiment name or delete all the existent results from {} folder to use this name".format(
            args.experiment_name
        )

        os.makedirs(experiment_name, "audio_segments")

        if write_audio_segments:
            assert (
                audio_directory is not None
            ), "If writing the audio segments, you need to provvide the embedding containing the original audio sources"

        results = {}
        for index, file in os.listdir(embedding_folder):
            root = embedding_folder
            file = os.path.join(root, file)

            emb = np.load(file)

            results[file] = self.model.predict(emb) > self.th

            if write_audio_segments:
                audio_segs, sr = self.segment_audio(
                    os.path.join("..", audio_directory, file[:-4] + self.ext),
                    results[file],
                )

                for index_seg, seg in enumerate(audio_segs):
                    write(
                        os.path.join(experiment_name, file + str(index_seg) + self.ext),
                        self.sr,
                        seg,
                    )

        with open(os.path.join(experiment_name, "results.pkl"), "wb") as f:
            pickle.dump(f)

        return results


if __name__ == "__main__":

    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write("error: %s\n" % message)
            self.print_help()
            sys.exit(2)

    parser = MyParser(
        description="Run training with parameters defined in the relative json file"
    )

    parser.add_argument(
        "--extract_embeddings",
        "-ee",
        action="store_true",
        help="If included, perform the embedding extraction before segmenting the audio files.",
    )

    parser.add_argument(
        "--embedding_folder",
        "-ef",
        type=str,
        required=True,
        help="The path to the directory storing the precomputed audio embeddings",
    )

    parser.add_argument(
        "--hyperparameter_file",
        "-hyp",
        type=str,
        required=True,
        help="Where to find the result file storing the various hyperparameters",
    )

    parser.add_argument(
        "--best_model_path",
        "-model",
        required=True,
        type=str,
        help="The name of the current experiment (the output will be saved in a folder with the same name)",
    )

    parser.add_argument(
        "--experiment_name",
        "-exp",
        default="new_experiment",
        type=str,
        help="The name of the current experiment (the output will be saved in a folder with the same name)",
    )

    parser.add_argument(
        "--batch_size",
        "-bs",
        default=1,
        type=int,
        help="the size of each mini batch during training",
    )

    parser.add_argument(
        "--num_gpus",
        "-gpus",
        default=1,
        type=int,
        help="Specify the number of gpus to use",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print out additional information during the training process",
    )

    parser.add_argument(
        "--audio_folder",
        "-af",
        type=str,
        required=True,
        help="The path to the directory storing the original audio files",
    )

    parser.add_argument(
        "--pca_reduce",
        "-pca",
        action="store_true",
        help="If included, use umap to reduce the embedding dimension to the dimension specified by --umap_value",
    )

    parser.add_argument(
        "--pca_value",
        "-pca_v",
        default=167,
        type=int,
        help="The number of components for umap reducer (see above)",
    )

    parser.add_argument(
        "--logistic_regression_baseline",
        "-lgr",
        action="store_true",
        help="Use logistic regression rather than recurrent neural network.",
    )

    parser.add_argument(
        "--uniform_interval",
        "-ui",
        default=1,
        type=float,
        help="the uniform interval from which the basic units were extracted",
    )

    parser.add_argument(
        "--adaptive_uniform",
        "-aus",
        action="store_true",
        help="if included, assume that adaptive uniform segmentation was used to extract the basic units, where each audio was divided into 100 equally sized portions",
    )

    parser.add_argument(
        "--threshold",
        "-th",
        type=float,
        default=0.5,
        help="The threshold that is applied to the system output and over which segmentation occurs. Increase to output less boundaries, decrease to output more.",
    )

    parser.add_argument(
        "--return_just_segmentation",
        "-rjs",
        action="store_false",
        help="if included, do not write to the output directory the segmented audio files as a result of the predict function",
    )

    args = parser.parse_args()

    if args.logistic_regression_baseline:
        predictor = LogReg_Predictor(
            args.best_model_path,
            adaptive_uniform_interval=args.adaptive_uniform,
            uniform_interval=args.uniform_interval,
        )
    else:
        predictor = Predictor(
            args.hyperparameter_file,
            args.best_model_path,
            args.pca_reduce,
            args.pca_value,
            threshold=args.threshold,
        )

    if args.extract_embeddings:
        predictor.create_embeddings(
            predictor.encoder,
            args.audio_folder,
            args.embedding_folder,
            args.uniform_interval,
            args.adaptive_uniform,
            args.verbose,
            True,
        )

        pooling_idx = predictor.encoder.find("_")
        if pooling_idx > -1:
            args.embedding_folder = os.path.join(
                args.embedding_folder, predictor.encoder[pooling_idx:]
            )

    results = predictor.predict(
        args.embedding_folder,
        args.experiment_name,
        write_audio_segments=args.return_just_segmentation,
        audio_directory=args.audio_folder,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        verbose=args.verbose,
    )
