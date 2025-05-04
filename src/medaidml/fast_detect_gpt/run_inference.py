# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import argparse
from medaidml.fast_detect_gpt.model import load_tokenizer, load_model
from scipy.stats import norm
from typing import Tuple
import pandas as pd
from tqdm import tqdm
import os

from medaidml import DATA_TEST_JSON, DATA_TRAIN_JSON, RESULTS_DIR
from medaidml.utils import json_to_dataframe, split_val_test, get_necessary_columns

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()

# Considering balanced classification that p(D0) equals to p(D1), we have
#   p(D1|x) = p(x|D1) / (p(x|D1) + p(x|D0))
def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob

class FastDetectGPT:
    def __init__(self, args):
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        if args.sampling_model_name != args.scoring_model_name:
            self.sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
            self.sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
            self.sampling_model.eval()
        # To obtain probability values that are easy for users to understand, we assume normal distributions
        # of the criteria and statistic the parameters on a group of dev samples. The normal distributions are defined
        # by mu0 and sigma0 for human texts and by mu1 and sigma1 for AI texts. We set sigma1 = 2 * sigma0 to
        # make sure of a wider coverage of potential AI texts.
        # Note: the probability could be high on both left side and right side of Normal(mu0, sigma0).
        #   gpt-j-6B_gpt-neo-2.7B: mu0: 0.2713, sigma0: 0.9366, mu1: 2.2334, sigma1: 1.8731, acc:0.8122
        #   gpt-neo-2.7B_gpt-neo-2.7B: mu0: -0.2489, sigma0: 0.9968, mu1: 1.8983, sigma1: 1.9935, acc:0.8222
        #   falcon-7b_falcon-7b-instruct: mu0: -0.0707, sigma0: 0.9520, mu1: 2.9306, sigma1: 1.9039, acc:0.8938
        distrib_params = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'gpt-neo-2.7B_gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983, 'sigma1': 1.9935},
            'falcon-7b_falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
        }
        key = f'{args.sampling_model_name}_{args.scoring_model_name}'
        self.classifier = distrib_params[key]

    # compute conditional probability curvature
    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.args.sampling_model_name == self.args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)

    # compute probability
    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob, crit, ntoken

def load_data(seed: int = 1, development: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # load data
    train_df = json_to_dataframe(DATA_TRAIN_JSON)
    no_dataleak_df = json_to_dataframe(DATA_TEST_JSON)
    # split data
    _, _, test_df = split_val_test(train_df, seed=seed, test_size=0.3)
    if development:
        test_df = test_df.sample(frac=0.1, random_state=seed)
        no_dataleak_df = no_dataleak_df.sample(frac=0.1, random_state=seed)
    # get necessary columns
    no_dataleak_df = get_necessary_columns(no_dataleak_df)
    test_df = get_necessary_columns(test_df)
    return test_df, no_dataleak_df

def predict_for_dataset(detector: FastDetectGPT, dataset: pd.DataFrame) -> pd.DataFrame:
    # get the text from the dataset
    # compute the probability for each text
    result = pd.DataFrame()
    for _, data in tqdm(dataset.iterrows(), desc="Predicting", total=len(dataset), unit="text"):
        prob, _, _ = detector.compute_prob(data['text'])
        result = pd.concat([result, pd.DataFrame([{'Ground Truth': data['target'],
                                                   'Prediction': int(prob > 0.5),
                                                   'language': data['language'],
                                                   'source': data['source']}])], ignore_index=True)
    return result

def run(args):
    detector = FastDetectGPT(args)
    # input text
    print('Running Inference on 5 splits...')
    print('')
    for i in range(5):
        print(f'Running Inference on split {i + 1}...')
        test_df, no_dataleak_df = load_data(seed=i+1, development=False)
        test_result = predict_for_dataset(detector, test_df)
        no_dataleak_result = predict_for_dataset(detector, no_dataleak_df)
        # save the result
        resulting_dir = os.path.join(RESULTS_DIR, "fast_detect_gpt", str(i+1))
        os.makedirs(resulting_dir, exist_ok=True)
        test_result.to_csv(os.path.join(resulting_dir, "results_test.csv"), index=False)
        no_dataleak_result.to_csv(os.path.join(resulting_dir, "results_no_dataleak.csv"), index=False)
    print('Inference completed!')   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_model_name', type=str, default="falcon-7b")
    parser.add_argument('--scoring_model_name', type=str, default="falcon-7b-instruct")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    run(args)