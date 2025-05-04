import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Union, List
import argparse
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift

from medaidml import RESULTS_DIR

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FFT analysis on data")
    parser.add_argument("--fft_method", type=str, default="fft", help="FFT method to use", choices=["fft", "periodogram"])
    parser.add_argument("--fft_preprocess_method", type=str, default="zscore", help="FFT preprocess method to use", choices=["zscore", "minmax", "log", "logzs"])
    parser.add_argument("--fft_value", type=str, default="norm", help="FFT value to use", choices=["norm", "real", "imag"])
    parser.add_argument("--require_sid", action='store_true', help="Append sequence id to output file")
    return parser.parse_args()

class FFTProcessor(object):
    def __init__(self, method, preprocess, value, require_sid, verbose=False):
        self.method = method
        self.preprocess = preprocess
        self.value = value
        self.require_sid = require_sid
        self.verbose = verbose
    
    def _read_data(self,
                   data_file: str,
                   N: int = np.inf) -> List[Tuple[np.ndarray, int, str, str]]:
        data = []
        with open(data_file, 'r') as f:
            count = 0
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                text, label, lang, source = line.split('\t')
                num = list(map(float, text.split()))
                data.append((num, int(label), lang, source))
                count += 1
                if count >= N:
                    break
        return data
    
    def _preprocess(self, input_data: List[Tuple[np.ndarray, int, str, str]]) -> List[Tuple[np.ndarray, int, str, str]]:
        data = input_data.copy()
        if self.preprocess == 'zscore':
            data_zs = []
            epsion = 1e-6
            for d, t, l, s in data:
                d = np.asarray(d)
                d_mean = np.mean(d)
                d_std = np.std(d)
                d_norm = (d - d_mean) / (d_std + epsion)
                data_zs.append((d_norm, t, l, s))
            data = data_zs.copy()
        elif self.preprocess == 'minmax':
            data_mm = []
            for d, t, l, s in data:
                d = np.asarray(d)
                d_min = np.min(d)
                d_max = np.max(d)
                d_norm = (d - d_min) / (d_max - d_min)
                data_mm.append((d_norm, t, l, s))
            data = data_mm.copy()
        elif self.preprocess == 'log':
            data_log = []
            for d, t, l, s in data:
                d = np.asarray(d)
                d_log = np.log(d + 1)
                data_log.append((d_norm, t, l, s))
            data = data_log.copy()
        elif self.preprocess == 'logzs':
            data_logzs = []
            epsion = 1e-6
            for d, t, l, s in data:
                d = np.asarray(d)
                d_log = np.log(d + 1)
                d_mean = np.mean(d_log)
                d_std = np.std(d_log)
                d_norm = (d_log - d_mean) / (d_std + epsion)
                data_logzs.append((d_norm, t, l, s))
            data = data_logzs.copy()
        elif self.preprocess != 'none':
            raise ValueError(f'Unknown preprocess method: {self.preprocess}. Please choose from [none, zscore, minmax, log, logzs].')
        return data
    
    def _create_input_df(self, data: List[Tuple[np.ndarray, int]], require_sid=True) -> pd.DataFrame:
        if require_sid:
            df = pd.DataFrame({
                'value': np.concatenate(d for d, _ in data),
                'sid': np.concatenate([np.repeat(i, len(d)) for i, d, _ in enumerate(data)]),
                'label': np.concatenate(t for _, t in data)
            })
        else:
            df = pd.DataFrame({
                'value': np.concatenate(d for d, _ in data),
                'label': np.concatenate(t for _, t in data)
            })
        return df

    def _periodogram_batch(self,
                           data: List[Tuple[np.ndarray, int, str, str]],
                           require_sid=False) -> Tuple[list, list, list, list, list, Optional[list]]:
        """
        Periodogram method (with smoothing window)
        """
        freqs, powers, labels, seq_ids, languages, sources = [], [], [], [], [], []
        for i in tqdm(range(len(data))):
            x, t, l, s = data[i]
            f, p = self._periodogram(x)
            freqs.append(f)
            powers.append(p)
            labels.append(np.repeat(t, len(f)))
            languages.append(np.repeat(l, len(f)))
            sources.append(np.repeat(s, len(f)))
            if require_sid:
                seq_ids.append(np.array([i] * len(f)))
        return freqs, powers, labels, languages, sources, seq_ids
    
    def _periodogram(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f, p = signal.periodogram(data)
        return f, p
    
    def _fft_batch(self,
                   data: List[Tuple[np.ndarray, int, str, str]],
                   require_sid=False,
                   verbose=False) -> Tuple[list, list, list, list, list, Optional[list]]:
        """
        FFT batch
        """
        freqs, powers, labels, languages, sources = [], [], [], [], []
        sids = [] if require_sid else None
        for i in tqdm(range(len(data)), disable = not verbose):
            x, t, l, s = data[i]
            try:
                f, p = self._fft(x)
            except Exception:
                print(f'Error in sample {i}: {x}')
                raise
            freqs.append(f)
            powers.append(p)
            labels.append(np.repeat(t, len(f)))
            languages.append(np.repeat(l, len(f)))
            sources.append(np.repeat(s, len(f)))
            if require_sid:
                sids.append(np.array([i] * len(f)))
        return freqs, powers, labels, languages, sources, sids

    def _fft(self, data: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
        """
        FFT
        """
        if isinstance(data, list):
            data = np.asarray(data)
        N = data.shape[-1]
        freq_x = fftshift(fftfreq(N))
        fft_res = fftshift(fft(data))
        if self.value == 'real':
            sp_x = fft_res.real
        elif self.value == 'imag':
            sp_x = fft_res.imag
        else:
            sp_x = np.abs(fft_res) # equivalent to np.sqrt(fft_res.real**2 + fft_res.imag**2)
        return freq_x[len(freq_x)//2:], sp_x[len(sp_x)//2:]
    
    def _create_fft_df(self,
                       freqs: list,
                       powers: list,
                       labels: list,
                       languages: list,
                       sources: list,
                       sids=None) -> pd.DataFrame:
        if sids is not None:
            df = pd.DataFrame.from_dict({
                'sid': np.concatenate(sids),
                'freq': np.concatenate(freqs),
                'power': np.concatenate(powers),
                'label': np.concatenate(labels),
                'language': np.concatenate(languages),
                'source': np.concatenate(sources)
            })
        else:
            df = pd.DataFrame.from_dict({
                'freq': np.concatenate(freqs),
                'power': np.concatenate(powers),
                'label': np.concatenate(labels),
                'language': np.concatenate(languages),
                'source': np.concatenate(sources)
            })
        return df
    
    def process(self, input_data: Union[str, list]) -> pd.DataFrame:
        """
        Carry out FFT analysis on data stored in input_file
        """
        if isinstance(input_data, str):
            data_list = self._read_data(input_data)
            data = [(np.asarray(d),t,l,s) for d, t, l, s in data_list]
        else:
            data = input_data.copy()

        # Preprocess
        data = self._preprocess(data)

        # Compute
        if self.method == 'periodogram':
            freqs, powers, labels, languages, sources, sids = self._periodogram_batch(data, require_sid=self.require_sid, verbose=self.verbose)
        elif self.method == 'fft':
            freqs, powers, labels, languages, sources, sids = self._fft_batch(data, require_sid=self.require_sid, verbose=self.verbose)
        else:
            raise ValueError(f'Unknown method: {self.method}. Please choose from [fft, periodogram].')

        # Collect result 
        df = self._create_fft_df(freqs, powers, labels, languages, sources, sids)

        return df
    
if __name__ == "__main__":
    args = get_args()
    FFT_METHOD = args.fft_method
    FFT_PREPROCESS_METHOD = args.fft_preprocess_method
    FFT_VALUE = args.fft_value
    REQUIRE_SID = args.require_sid 
    
    out_dir = os.path.join(RESULTS_DIR, "fourier_gpt", "fft_transformed")
    in_dir = os.path.join(RESULTS_DIR, "fourier_gpt", "likelihood_scores")
    os.makedirs(out_dir, exist_ok=True)
    
    fft_processor = FFTProcessor(method=FFT_METHOD, 
                             preprocess=FFT_PREPROCESS_METHOD, 
                             value=FFT_VALUE, 
                             require_sid=REQUIRE_SID,
                             verbose=True)
    df_train = fft_processor.process(os.path.join(in_dir, "nll_train"))
    df_train.to_csv(os.path.join(out_dir, "fft_train.csv"), index=False)
    
    df_test = fft_processor.process(os.path.join(in_dir, "nll_test"))
    df_test.to_csv(os.path.join(out_dir, "fft_test.csv"), index=False)
    
    print(f"FFT analysis completed. Results saved to {out_dir}.")