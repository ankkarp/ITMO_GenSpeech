import math
import time
import heapq
from pathlib import Path
from typing import List, Tuple, Union

import kenlm
import torch
import torchaudio
import Levenshtein
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-960h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0
        ):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    def greedy_decode(self, logits: torch.Tensor) -> str:
        pred_idxs = torch.argmax(logits, dim=-1)
        return self._seq_to_transcript(pred_idxs.tolist())

    def _seq_to_transcript(self, seq):
        tokens = []
        prev_token = None
        for idx in seq:
            if idx != self.blank_token_id and idx != prev_token:
                tokens.append(self.vocab[idx])
            prev_token = idx
        return ''.join(tokens).replace(self.word_delimiter, ' ').strip()

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        log_probs = F.log_softmax(logits, dim=-1)
        T, V = log_probs.shape

        beams = [([], 0.0)]  # (sequence, score)

        for t in range(T):
            new_beams = []
            for seq, score in beams:
                for v in range(V):
                    new_seq = seq + [v]
                    new_score = score + log_probs[t, v].item()
                    new_beams.append((new_seq, new_score))
            beams = heapq.nlargest(self.beam_width, new_beams, key=lambda x: x[1])

        if return_beams:
            return beams
        else:
            return self._seq_to_transcript(beams[0][0])

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        log_probs = F.log_softmax(logits, dim=-1)
        T, V = log_probs.shape

        beams = [([], 0.0, "<s>")]  # (sequence, score, LM state)

        for t in range(T):
            new_beams = []
            for seq, score, lm_prefix in beams:
                for v in range(V):
                    new_seq = seq + [v]
                    token = self.vocab[v]
                    new_score = score + log_probs[t, v].item()

                    # Apply LM shallow fusion
                    if token != self.word_delimiter and token != self.processor.tokenizer.pad_token:
                        lm_state = lm_prefix + token
                        lm_score = self.lm_model.score(lm_state, bos=True, eos=False)
                        new_score += self.alpha * lm_score + self.beta
                        new_beams.append((new_seq, new_score, lm_state))
                    else:
                        new_beams.append((new_seq, new_score, lm_prefix))

            beams = heapq.nlargest(self.beam_width, new_beams, key=lambda x: x[1])

        return self._seq_to_transcript(beams[0][0])

    def lm_rescore(self, beams: List[Tuple[str, float]]) -> str:
        rescored = []
        for seq, score in beams:
            transcript = self._seq_to_transcript(seq)
            words = transcript.split()
            lm_score = self.lm_model.score(" ".join(words), bos=True, eos=True)
            total_score = self.alpha * lm_score + self.beta * len(words) + score
            rescored.append((transcript, total_score))
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[0][0]

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")


def test(decoder, audio_path, true_transcription):

    res_dict = {}
    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding")
        t0 = time.perf_counter()
        transcript = decoder.decode(audio_input, method=d_strategy).strip()
        res_dict[f'{d_strategy}_t'] = time.perf_counter() - t0
        print(f"{transcript}")
        score = Levenshtein.distance(true_transcription, transcript) / max(len(true_transcription), len(transcript))
        print(f"Character-level Levenshtein distance: {score}")
        res_dict[f'{d_strategy}_levenshtein'] = score
    return res_dict


def test(decoder, audio_path, true_transcription):

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding") 
        transcript = decoder.decode(audio_input, method=d_strategy)
        print(f"{transcript}")
        print(f"Character-level Levenshtein distance: {Levenshtein.distance(true_transcription, transcript.strip())}")
        print(f"Normalized Character-level Levenshtein distance: {Levenshtein.distance(true_transcription, transcript.strip()) / max(len(transcript.strip()), len(true_transcription))}")


def score(decoder, audio_input, true_transcription, d_strategies, lm_model_name=None):
    res_records = []
    try:
        for d_strategy in d_strategies:
            res_dict = {'method': d_strategy, 'beam_width': decoder.beam_width,
                        'alpha': decoder.alpha, 'beta': decoder.beta,
                        'lm_model': lm_model_name, 'input': true_transcription}
            t0 = time.perf_counter()
            transcript = decoder.decode(audio_input, method=d_strategy)
            res_dict[f't'] = time.perf_counter() - t0
            res_dict[f'transcript'] = transcript
            score = Levenshtein.distance(true_transcription, transcript.strip())  
            norm_score = score / max(len(true_transcription), len(transcript))
            res_dict[f'levenshtein'] = score
            res_dict[f'levenshtein_norm'] = norm_score
            res_records.append(res_dict)
        print(res_records)
    except Exception as e:
        print(e)
    finally:
        return res_records

if __name__ == "__main__":
    
    test_samples = [
        ("examples/sample1.wav", "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("examples/sample2.wav", "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
        ("examples/sample3.wav", "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("examples/sample4.wav", "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
        ("examples/sample5.wav", "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES"),
        ("examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
        ("examples/sample7.wav", "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS"),
        ("examples/sample8.wav", "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE"),
    ]



    # model_paths = list(filter(lambda x: x.is_file(), Path('lm').glob('*')))
    # beam_width_grid = list(np.linspace(1, 5, 5, dtype=int))
    # ratio_grid = list(np.linspace(0.1, 1, 10, dtype=float))
    # scale_grid = list(np.linspace(1, 3, 3, dtype=int))
    # pbar = tqdm(total=
                # len(model_paths) * len(t est_samples) + \
                # len(model_paths) * len(test_samples) * len(beam_width_grid) + \
                    # len(model_paths) * len(test_samples) * len(beam_width_grid) * len(ratio_grid) * len(scale_grid)
                # )
    # try:
    #     score_records = []
    #     for model_path in model_paths:
    #         decoder = Wav2Vec2Decoder(lm_model_path=str(model_path), beam_width=1, alpha=0.1, beta=0.9)
    #         for audio_path, target in test_samples:
    #             audio_input, sr = torchaudio.load(audio_path)

    #             # score_records.extend(score(decoder, audio_input, target, ["greedy", "beam", "beam_lm", "beam_lm_rescore"], model_path.stem))
    #             pbar.update(4)
    #             for beam_width in [beam_width_grid[0]]:
    #                 decoder.beam_width = beam_width
    #                 # score_records.extend(score(decoder, audio_input, target, ['beam', "beam_lm", "beam_lm_rescore"], model_path.stem))
    #                 pbar.update(3)
    #                 for ratio in ratio_grid:
    #                     for scale in scale_grid:
    #                         decoder.alpha = round(ratio * scale, 1)
    #                         decoder.beta = round(scale - decoder.alpha, 1)
    #                         score_records.extend(score(decoder, audio_input, target, ["beam_lm", "beam_lm_rescore"], model_path.stem))
    #                         pbar.update(2)
                    
    #     pd.DataFrame.from_records(score_records).to_csv(f'data/{model_path.stem}_scores3.csv')
    # except (Exception, KeyboardInterrupt) as e:
    #     print(e)
    # finally:
    #     pd.DataFrame.from_records(score_records).to_csv('scores.csv')



    decoder = Wav2Vec2Decoder()

    _ = [test(decoder, audio_path, target) for audio_path, target in test_samples]
