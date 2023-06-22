# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:37:53 2021

@author: User
"""

import os
import librosa
import soundfile as sf
import argparse
import numpy as np
import re
import pickle
import json
import sys
import torch
from speechbrain.pretrained import EncoderClassifier
# from inaSpeechSegmenter import Segmenter
from transformers import Wav2Vec2Processor, Wav2Vec2Model
#from extract_acoustic_features import get_acoustic_features
from librosa import yin
from speechbrain.pretrained import VAD
import openl3



def create_vad_segments(segmentation, lab_times, vad = True, speechbrain = False,
                        append_labs = False):
    
    end_index = 1 if speechbrain else 2
    
    if vad:
        index = 0
            
        segments = []
        labs = []
        
        for time in lab_times:
            segment = []
            for seg in segmentation[index:]:
                index += 1
                segment.append(seg)
                if float(time[1])<seg[end_index]:
                    if segment:
                        break
            
            segments.append(segment)
            if append_labs:
                if len(segment)-1>0:
                    labs.append([0 for x in range(len(segment)-1)]+[1])
            else:
                labs.extend([0 for x in range(len(segment)-1)]+[1])
        deleted = 0
        
        if not append_labs:
            clean_segments = []
            for index_seg, seg in enumerate(segments):
                if not seg:
                    labs.pop(index_seg-deleted)
                    deleted+=1
                else:
                    clean_segments.append(seg)
        else:
            clean_segments = segments
        
    if append_labs:
        if len(segments[-1])>len(labs[-1]):
                labs[-1].extend([0 for x in range(len(segmentation[index:]))])
    elif len(segmentation)>len(labs):
        labs.extend([0 for x in range(len(segmentation[index:]))])
        labs[-1] = 1
    else:
        pass
    
    return clean_segments, labs

def create_uniform_segments(lab_times, segment_duration = 1, append_labs = False):
   
   segments = []
   labs = []
   previous_time = 0     
   for time in lab_times:
       diff = round(float(time[1])) - previous_time
       tot_segments = diff/segment_duration
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
       
       segments.extend([(previous_time + segment_duration*i, previous_time + segment_duration*(i+1)) for i in range(round(tot_segments))])
       
       previous_time = round(float(time[1]))
       
   return segments, labs


def to_sample(sample_rate, time):
    return int(sample_rate*time)

def to_time(sample_rate, samples):
    return samples/sample_rate

def main(args):
    
    if args.vad:
        if args.speechbrain:
            seg = VAD.from_hparams(source="SpeechBrainVAD")
            # seg = VAD.from_hparams(source =)
        else:
            seg = Segmenter()
    
    verbose = args.verbose
    all_labs_dictionary = {}
    if not os.path.exists(args.out_directory):
        os.makedirs(os.path.join(args.out_directory))
        existent_files = []
    else:
        if args.openl3 or args.wav2vec:
            existent_files = os.listdir(os.path.join(args.out_directory, "_mean"))
            # print(existent_files)
        else:
            existent_files = os.listdir(args.out_directory)
        
        if args.continue_from_check:
            print('Warning: the directory where to store the results already exists! Embeddings from the same files will be skipped...')
        else:
            print('Warning: the directory where to store the results already exists! Embeddings will be saved there: this might overwrite existent files...')
    
    if args.ecapa:
        if verbose:
            print('Computing ecapa embeddings, rather than xvectors...')
        model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="dehdeh/spkrec-ecapa-voxceleb")
    elif args.openl3:
        class encoder:
            def __init__(self):
                input_repr, content_type, embedding_size = 'mel128', 'env', 512
                self.model = openl3.models.load_audio_embedding_model(input_repr, content_type, embedding_size)
            def encode_batch(self, audio):
                embs, ts = openl3.get_audio_embedding(audio, 16000, model = self.model, verbose = False)
                return embs
        
        model = encoder()
    
    elif args.prosodic_feats:
        if verbose:
            print("Using mel-filterbank energies and prosodic feature statistics to generate embeddings.")
        class encoder:
            def encode_batch(self, audio, previous_pitches = None):
                return get_acoustic_features(audio, 16000, previous_pitches)
        
        model = encoder()
    
    elif args.mfcc:
        if verbose:
            print("Using mel-cepstral coefficients statistics to generate embeddings.")
        class encoder:
            def encode_batch(self, audio):
                return get_acoustic_features(audio, 16000, mfcc = True)
        
        model = encoder()
    
    elif args.wav2vec:
        class encoder:
            def __init__(self):
                self.preprocessor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
                
            def encode_batch(self, audio):
                embs = self.preprocessor(audio, sampling_rate = 16000, return_tensors='pt').input_values
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
        model = EncoderClassifier.from_hparams(source = 'pretrained_models/', hparams_file = 'hyperparams.yaml', savedir="dehdeh/spkrec-xvect-voxceleb")
    
    data = []
    times = []
    
    
    file_paths = []
    audio_paths = []
    filenames = []
    
    for root, directory, files in os.walk(args.audio_directory):
        for file in files:
            if file.endswith('mp3') or file.endswith('wav'):
                filename = re.findall("(.+)\\.\\w+$", file)[-1]
                # print(filename)
            else:
                continue
            filenames.append(filename)
            
            if filename:
                file_pattern = '\\s?({}\\S*)'.format(filename)
                # print(file_pattern)
                
                transcript_name = re.findall(file_pattern, ' '.join(os.listdir(args.data_directory)))[-1]
                
                file_paths.append(os.path.join(args.data_directory, transcript_name))
                # file_paths.append(os.path.join(args.data_directory, filename[-1]+'.pkl'))
                audio_paths.append(os.path.join(args.audio_directory, file))
    
    all_segments = []
    all_labs = []
    lab_index = 0
    
    if args.extract_labels:
        if args.BMAT:
            print(audio_paths)
            with open(args.lab_file) as f:
                lab_file = json.load(f)
        else:
            lab_file = np.load(args.lab_file)
    
    if args.BMAT:
        for k,v in lab_file.items():
            times.append(v)
        
    else:
        for file_path in file_paths:
            if os.stat(file_path).st_size:
                if file_path.endswith('pkl'):
                    with open(file_path, 'rb') as f:
                        
                        time_stamped_sentences = pickle.load(f)
                elif file_path.endswith('json'):
                    with open(file_path) as f:
                        
                        time_stamped_sentences = json.load(f)
                        
                else:
                    raise ValueError('The extension of the provided timestamped sentences need to be in json or pkl format!')
            
            sentences = []
            time_stamps = []
            
            for t in time_stamped_sentences:
                sentences.append(t['sentence'])
                time_stamps.append((t['start'], t['end']))
            
            data.append(sentences)
            times.append(time_stamps)
    
    if verbose:
        print('Starting audio embedding extraction...')
    
    for index, timestamps in enumerate(times):
        if args.BMAT:
            print()
            timestamps = lab_file[os.path.basename(audio_paths[index])[:-4]]
        if args.continue_from_check and existent_files:
            current_file = os.path.basename(audio_paths[index])[:-4]
            file_exists = re.findall(current_file, " ".join(existent_files))
            # print(file_exists)
            
            if file_exists:
                
                for time in timestamps:
                    lab_index += 1
                
                print("File {} exists in target directory: skipping this file".format(file_exists[0] + '.npy'))
                continue
                
        
        read_audio = True
        
        if args.vad:
            if verbose and args.speechbrain:
                print("Segmenting with Speechbrain's VAD...")
            elif verbose:
                print('Segmenting with INA SpeechSegmenter...')
            
            try:
                if args.speechbrain:
                    read_audio = False
                    audio, sr = librosa.load(audio_paths[index])
                    audio = librosa.resample(audio, sr, 16000)
                    
                    temp_file = sf.write('placeholder.wav', audio, 16000)
                    try:
                        segmentation = seg.get_speech_segments('placeholder.wav', apply_energy_VAD= args.postprocess)
                    except RuntimeError:
                        if verbose:
                            print('Warning: Postprocessing failed... trying with just neural VAD.')
                        segmentation = seg.get_speech_segments('placeholder.wav')
                    os.remove('placeholder.wav')
                    sr = 16000
                
                else:
                    segmentation = seg(audio_paths[index])
            except MemoryError:
                read_audio = False
                
                audio, sr = librosa.load(audio_paths[index])
    
                quarter_length = len(audio)//4
                
                partial_segs = []
                
                for i in range(3):
                    start = quarter_length*i
                    end = quarter_length*(i+1)
                    temp_file = sf.write('placeholder.wav', audio[start:end], sr)
                    if args.speechbrain:
                        if i>0:
                            segment = seg.get_speech_segments('placeholder.wav', apply_energy_VAD= args.postprocess)
                            partial_segs.append(segment+to_time(sr, quarter_length*i))
                        else:
                            partial_segs.append(seg.get_speech_segments('placeholder.wav', apply_energy_VAD= args.postprocess))
                    else:
                        partial_segs.append(seg('placeholder.wav'))
                
                temp_file = sf.write('placeholder.wav', audio[end:], sr)
                if args.speechbrain:
                        segment = seg.get_speech_segments('placeholder.wav', apply_energy_VAD= args.postprocess)
                        partial_segs.append(segment+to_time(sr, quarter_length*3))
                else:
                    partial_segs.append(seg('placeholder.wav'))
                    
                segmentation = []
                
                
                if args.speechbrain:
                    new_segmentation = []
                    
                    segmentation = torch.cat(partial_segs, axis=0)
                    for current_index, time in enumerate(segmentation):
                        if current_index%2:
                            start = time
                        else:
                            new_segmentation.append(('segment', start, time))
                            
                    segmentation = new_segmentation
                
                else:
                    current_index = 0
                    
                    for partial_segment in partial_segs:
                        for seg_index, segment in enumerate(partial_segment):
                            start_time = segment[1]+current_index
                            end_time = segment[2]+current_index
                            segmentation.append((segment[0], start_time, end_time))
                        
                        current_index = end_time
                
                os.remove('placeholder.wav')
                
            
        
                if verbose:
                    print('Done with Segmentation...')
                if args.just_speakers and not args.speechbrain:
                    new_segmentation = []
                    for segment in segmentation:
                        if re.findall('male', segment[0]):
                            new_segmentation.append(segment)
                    
                    segmentation = new_segmentation
                    
            if args.extract_labels:
                lab_time = []
                
                if args.BMAT:
                    for time in timestamps:
                        if lab_time:
                            lab_time.append((lab_time[-1][1], lab_time[-1][1]+time))
                        else:
                            lab_time.append((0, time))
                else:
                    
                    for time in timestamps:
                        if lab_file[lab_index]:
                            lab_time.append(time)
                        lab_index += 1
                        
                # filtering out files that are too long
                if re.findall("(24580|25539|25684|26071|26214|26321|26427)", audio_paths[index]):
                    continue
            
            
                segments, labs = create_vad_segments(segmentation, lab_time, speechbrain=args.speechbrain)
            
            
            if len(segmentation)!=len(labs):
                error = {'file_name':audio_paths[index],'segmentation':segmentation, 'labs':labs, 'lab_time':lab_time}
                with open('error.pkl', 'wb') as f:
                    pickle.dump(error, f)
                1/0
            # else:
            #     continue
            
            assert len(segmentation)==len(labs), 'Labs and segmentation lengths differ!'
            
            all_segments.append(segments)
            if args.concatenate_labels:
                all_labs.extend(labs)
            else:
                all_labs.append(labs)
            all_labs_dictionary[audio_paths[index]] = labs
        
        elif args.use_sentence_time:
             if verbose:
                print("Using sentence-level times, as extracted by the ASR tool...")
            
             labs = []
             for time in timestamps:
                if lab_file[lab_index]:
                    labs.append(1)
                else:
                    labs.append(0)
                lab_index += 1
            
             segmentation = timestamps
             if args.concatenate_labels:
                all_labs.extend(labs)
             else:
                all_labs.append(labs)
             all_labs_dictionary[audio_paths[index]] = labs
        
        else:
            if verbose:
                print("Doing uniform segmentaiton with segment duration = {}...".format(args.uniform_interval))
            
            lab_time = []
            # for time in timestamps:
            #     if lab_file[lab_index]:
            #         lab_time.append(time)
            #     lab_index += 1
            
            if args.BMAT:
                for time in timestamps:
                    if lab_time:
                        lab_time.append((lab_time[-1][1], lab_time[-1][1]+time))
                    else:
                        lab_time.append((0, time))
            else:
                
                for time in timestamps:
                    if lab_file[lab_index]:
                        lab_time.append(time)
                    lab_index += 1
            
            if re.findall("(24580|25539|25684|26071|26214|26321|26427)", audio_paths[index]):
                    continue
            
            if args.adaptive_uniform_segmentation:
                try:
                    segment_duration = lab_time[-1][1]/100
                except TypeError:
                    segment_duration = float(lab_time[-1][1])/100
            else:
                segment_duration = args.uniform_interval
            
            segmentation, labs = create_uniform_segments(lab_time, segment_duration = segment_duration, append_labs = args.concatenate_labels)
            
            if args.concatenate_labels:
                all_labs.extend(labs)
            else:
                all_labs.append(labs)
            all_labs_dictionary[audio_paths[index]] = labs
            
            assert len(segmentation)==len(labs), "Segmentation must be the same length as labels!"
        
        if verbose:
            print('Extracting audio embeddings for file {}'.format(audio_paths[index]))
        
        if read_audio:
            audio, sr = librosa.load(audio_paths[index])
        
        if sr!=16000:
            if verbose:
                print('Resampling audio to 16000 Hz...')
            audio = librosa.resample(audio, sr, 16000)
        
        audio_embeddings = []
        start_index = 0 if args.speechbrain or not args.vad else 1
        end_index = start_index + 1
        
        prev_pitches = None
        
        for index2, time in enumerate(segmentation):
            
            start = to_sample(16000, float(time[start_index]))
            
            # end = to_sample(16000, float(time[2]))
            try:
                end = to_sample(16000, float(segmentation[index2+1][start_index]))
                assert segmentation[index2+1][start_index]==time[1]
            except IndexError:
                end = to_sample(16000, float(time[end_index]))
            
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
                        signal = audio[start:start+(end-start)//4]
                        embeddings1 = model.encode_batch(signal).squeeze().squeeze()
                        signal2 = audio[start+(end-start)//4:start+(end-start)//2]
                        embeddings2 = model.encode_batch(signal2).squeeze().squeeze()
                        signal3 = audio[start+(end-start)//2:end-(end-start)//4]
                        embeddings3 = model.encode_batch(signal3).squeeze().squeeze()
                        signal4 = audio[end-(end-start)//4:end]
                        embeddings4 = model.encode_batch(signal3).squeeze().squeeze()
                        embeddings = (embeddings1 + embeddings2 + embeddings3 + embeddings4)/4
                    else:
                        # signal = torch.from_numpy(audio[start:start+(end-start)//6])
                        # embeddings1 = model.encode_batch(signal).squeeze().squeeze()
                        # signal2 = torch.from_numpy(audio[start+(end-start)//6:start+(end-start)//3])
                        # embeddings2 = model.encode_batch(signal2).squeeze().squeeze()
                        # signal3 = torch.from_numpy(audio[start+(end-start)//3:end-(end-start)//2])
                        # embeddings3 = model.encode_batch(signal3).squeeze().squeeze()
                        # signal4 = torch.from_numpy(audio[end-(end-start)//2:end-(end-start)//3])
                        # embeddings4 = model.encode_batch(signal4).squeeze().squeeze()
                        # signal5 = torch.from_numpy(audio[end-(end-start)//3:end-(end-start)//6])
                        # embeddings5 = model.encode_batch(signal5).squeeze().squeeze()
                        # signal6 = torch.from_numpy(audio[end-(end-start)//6:end])
                        # embeddings6 = model.encode_batch(signal6).squeeze().squeeze()
                        # embeddings = (embeddings1 + embeddings2 + embeddings3 + embeddings4 + embeddings5 + embeddings6)/6
                        
                        signal = torch.from_numpy(audio[start:start+(end-start)//12])
                        embeddings1 = model.encode_batch(signal).squeeze().squeeze()
                        signal2 = torch.from_numpy(audio[start+(end-start)//12:start+(end-start)//6])
                        embeddings2 = model.encode_batch(signal2).squeeze().squeeze()
                        signal3 = torch.from_numpy(audio[start+(end-start)//6:start + (end-start)//4])
                        embeddings3 = model.encode_batch(signal3).squeeze().squeeze()
                        signal4 = torch.from_numpy(audio[start+(end-start)//4:start+(end-start)//3])
                        embeddings4 = model.encode_batch(signal4).squeeze().squeeze()
                        signal5 = torch.from_numpy(audio[start+(end-start)//3:start+(end-start)//12*5])
                        embeddings5 = model.encode_batch(signal5).squeeze().squeeze()
                        signal6 = torch.from_numpy(audio[start+(end-start)//12*5:end-(end-start)//2])
                        embeddings6 = model.encode_batch(signal6).squeeze().squeeze()
                        signal7 = torch.from_numpy(audio[end-(end-start)//2:end-(end-start)//12*5])
                        embeddings7 = model.encode_batch(signal7).squeeze().squeeze()
                        signal8 = torch.from_numpy(audio[end-(end-start)//12*5:end-(end-start)//3])
                        embeddings8 = model.encode_batch(signal8).squeeze().squeeze()
                        signal9 = torch.from_numpy(audio[end-(end-start)//3:end-(end-start)//4])
                        embeddings9 = model.encode_batch(signal9).squeeze().squeeze()
                        signal10 = torch.from_numpy(audio[end-(end-start)//4:end-(end-start)//6])
                        embeddings10 = model.encode_batch(signal10).squeeze().squeeze()
                        signal11 = torch.from_numpy(audio[end-(end-start)//6:end-(end-start)//12])
                        embeddings11 = model.encode_batch(signal11).squeeze().squeeze()
                        signal12 = torch.from_numpy(audio[end-(end-start)//12:end])
                        embeddings12 = model.encode_batch(signal12).squeeze().squeeze()
                        embeddings = (embeddings1 + embeddings2 + embeddings3 + embeddings4 + embeddings5 + embeddings6 + embeddings7 + embeddings8 + embeddings9 + embeddings10 + embeddings11 + embeddings12)/12
            
                if args.ecapa:
                    assert embeddings.shape[0] == 192, 'NOOOOO'
                    
                else:
                    if args.openl3:
                        if len(embeddings.shape)==1:
                            embeddings = embeddings.reshape(1, 512)
                        assert embeddings.shape[1] == 512, 'NOOOOO'
                    elif args.wav2vec:
                        if len(embeddings.shape)==1:
                            embeddings = embeddings.reshape(1, 768)
                        assert embeddings.shape[1] == 768, 'NOOOOO'
                    elif args.CREPE:
                        if len(embeddings.shape)==1:
                            embeddings = embeddings.reshape(1, 256)
                        assert embeddings.shape[1] == 256, 'NOOOOO'
                    elif args.mfcc:
                        if len(embeddings.shape)==1:
                            embeddings = embeddings.reshape(1, 200)
                        assert embeddings.shape[1] == 200, 'NOOOOO'
                    else:
                        assert embeddings.shape[0] == 512, 'NOOOOO'
            
            if args.openl3 or args.wav2vec or args.CREPE:
                # if args.max:
                #     mean_embedding = np.max(embeddings, axis = 0)
                # elif args.gap_sentence:
                #     mean_embedding = embeddings[-1]
                # else:
                #     mean_embedding = embeddings.mean(axis=0)
                # if args.add_std:
                #     mean_embedding = np.concatenate((mean_embedding, np.std(embeddings, axis = 0)))
                #     assert mean_embedding.shape[1] == 1024, 'NOOOOO'
                
                audio_embeddings.append(embeddings)
                
            elif args.prosodic_feats:
                audio_embeddings.append(mean_embedding)
            
            elif args.mfcc:
                audio_embeddings.append(embeddings)
            
            else:
                mean_embedding = embeddings.detach().numpy()
            
                audio_embeddings.append(mean_embedding)
            
            
        
        audio_embeddings = np.array(audio_embeddings)
        
        assert len(audio_embeddings)==len(segmentation), 'Something went wrong!'
        
        out_file = os.path.join(args.out_directory, filenames[index])
        
        if verbose:
            print('Embeddings extracted!\nWriting them to {}'.format(out_file))
        
        if args.openl3 or args.wav2vec or args.CREPE:
            if not os.path.exists(os.path.join(args.out_directory, '_mean')):
                os.mkdir(os.path.join(args.out_directory, '_mean'))
                os.mkdir(os.path.join(args.out_directory, '_max'))
                os.mkdir(os.path.join(args.out_directory, '_no_reduction'))
                os.mkdir(os.path.join(args.out_directory, '_mean_std'))
                os.mkdir(os.path.join(args.out_directory, '_max_std'))
                os.mkdir(os.path.join(args.out_directory, '_last'))
                os.mkdir(os.path.join(args.out_directory, '_delta_gap'))
            
            with open(os.path.join(args.out_directory, '_no_reduction', filenames[index])+'.pkl', 'wb') as f:
                pickle.dump(audio_embeddings, f)
            np.save(os.path.join(args.out_directory, '_mean', filenames[index]), np.array([e.mean(axis=0) for e in audio_embeddings]))
            np.save(os.path.join(args.out_directory, '_max', filenames[index]), np.array([np.max(e, axis = 0) for e in audio_embeddings]))
            np.save(os.path.join(args.out_directory, '_mean_std', filenames[index]), np.array([np.concatenate((e.mean(axis=0), np.std(e, axis = 0))) for e in audio_embeddings]))
            np.save(os.path.join(args.out_directory, '_max_std', filenames[index]), np.array([np.concatenate((np.max(e, axis = 0), np.std(e, axis = 0))) for e in audio_embeddings]))
            np.save(os.path.join(args.out_directory, '_last', filenames[index]), np.array([e[-1] for e in audio_embeddings]))
            delta_gap = []
            for index_e, e in enumerate(audio_embeddings):
                try:
                    delta_gap.append(audio_embeddings[index_e+1][0]-e[-1])
                except IndexError:
                    delta_gap.append(e[-1])
            np.save(os.path.join(args.out_directory, '_delta_gap', filenames[index]), np.array(delta_gap))
                    
            
        else:
            np.save(out_file, audio_embeddings)
    
    if not os.path.exists(args.lab_out_dir):
        os.mkdir(args.lab_out_dir)
        
    if args.extract_labels:
        
        segment_file = os.path.join(args.lab_out_dir, "segments.pkl")
        
        labs_dictionary_file = os.path.join(args.lab_out_dir, "labs_dict.pkl")
        
        lab_out_file = os.path.join(args.lab_out_dir, 'labels')
        
        with open(segment_file, "wb") as fp:   #Pickling
            pickle.dump(all_segments, fp)
        
        with open(labs_dictionary_file, "wb") as fp:
            pickle.dump(all_labs_dictionary, fp)
        
        np.save(lab_out_file, np.array(all_labs))

if __name__=='__main__':
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
            
    
    parser = MyParser(
            description = 'Compute audio embeddings and store them in the specified directory')
    
    parser.add_argument('--data_directory', '-data', type=str,
                        help='directory containing the time stamped sentences to be segmented')
    
    parser.add_argument('--audio_directory', '-audio', type=str,
                        help='directory containing the audio to be segmented')
    
    parser.add_argument('--out_directory', '-od', default='results', type=str,
                        help='the directory where to store the segmented texts')
    
    parser.add_argument('--ecapa', '-e', action = 'store_true',
                        help = 'Compute ecapa embeddings instead of xvectors.')
    
    parser.add_argument('--verbose', '-vb', action='store_true', help='Whether to print messages during running.')
    
    parser.add_argument('--just_speakers', '-js', action='store_false', help = 'whether to exclude all detected audio that is not a speaker for computing xvectors')
    
    parser.add_argument('--extract_labels', '-exl', action = 'store_false', help = 'whether to also extract the labels relating the new segmentation and the sentence-level one')
    
    parser.add_argument('--lab_file', '-lab', default = 'nltk_podcast_labs.npy', type = str,
                        help = 'whether to also extract the labels relating the new segmentation and the sentence-level one')
    
    parser.add_argument('--lab_out_dir', '-lod', default = 'INA_podcast_segments', type = str,
                        help = 'The directory where to store the segments from INA Speech Segmenter and the relative labs for the BBC podcast dataset')
    
    parser.add_argument('--vad', '-vd', action = 'store_false', help = 'If included use uniform segmentation rather than a VAD engine to obtain the audio segments')
    
    parser.add_argument('--speechbrain', '-sb', action = 'store_true', help = 'If included it uses speechbrain rather than INA for audio segmentation (recommended)')

    parser.add_argument('--concatenate_labels', '-cl', action = 'store_true', help = 'If included, the new ground truth labels will be concatenated in a single array, rather than keeping the division between documents.')    

    parser.add_argument('--postprocess', '-pp', action = 'store_false', help = 'If included it disactivates the default postprocessing step of the VAD model from speechbrain.')
    
    parser.add_argument('--uniform_interval', '-ui', type = float, default = 1.0,
                        help = 'If using uniform segmentation, this argument specifies the time frame of each segment.')
    
    parser.add_argument('--use_sentence_time', '-ust', action = 'store_true',
                        help = 'If included, directly use the information from the timestamped sentences to segment the audio at the sentence level.')
    
    parser.add_argument('--openl3', action = 'store_true', help = 'use openl3 to extract audio embeddings')
    
    parser.add_argument('--wav2vec', action = 'store_true', help = 'use wav2vec2 to extract audio embeddings')
    
    parser.add_argument('--CREPE', action = 'store_true', help = 'use CREPE to extract audio embeddings')
    
    parser.add_argument('--prosodic_feats', action = 'store_true', help = 'use prosodic features as audio embeddings')
    
    parser.add_argument('--mfcc', action = 'store_true', help = 'use mfccs as audio embeddings')
    
    parser.add_argument('--max', action = 'store_true', help = 'if using openl3 or wav2vec, use the max operation to summarise the embeddings in a window, rather than the arithmetic mean.')
    
    parser.add_argument('--add_std', action = 'store_true', help = 'if using openl3 or wav2vec, add the standard deviation statistics to the aggregated embeddings')
    
    parser.add_argument('--gap_sentence', '-gs', action = 'store_true', help = 'Use just the last extracted embedding per sentence/segment instead of averaging all of them.')
    
    parser.add_argument('--continue_from_check', '-cont', action = 'store_true', help = 'Continue from previous experiment by avoiding extracting embeddings for files already present in the output folder.')
    
    parser.add_argument('--BMAT', action = 'store_true', help = 'If using BMAT, then the timing for generating the true labels is used differently (timing is not extracted from sentences, but from the true length of each segment composing the document).')
    
    parser.add_argument('--adaptive_uniform_segmentation', '-aus', action = 'store_true', help = 'Using uniform segmentation, where the length of each segment will be a hundredth of the total document length')
    
    args = parser.parse_args()
    
    main(args)
