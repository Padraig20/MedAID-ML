import os
import argparse
from tqdm import tqdm
from typing import Optional, Tuple
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from einops import rearrange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from medaidml import RESULTS_DIR, DATA_TRAIN_JSON, DATA_TEST_JSON
from medaidml.utils import json_to_dataframe

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run likelihood score estimations with GPT-2")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--model_path", type=str, default="", help="Path to the model")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--development", action="store_true", help="Use a smaller sample for development")
    return parser.parse_args()

def load_model(model_path: Optional[str],
               model: Optional[str]) -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    if model_path:
        model_path = model_path
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer(tokenizer_file=os.path.join(model_path, 'tokenizer.json'), 
                                  vocab_file=os.path.join(model_path, 'vocab.json'),
                                  merges_file=os.path.join(model_path, 'merges.txt'))
    else:
        model_path = model
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps') # for mac
    else:
        device = torch.device('cpu')
    model.to(device)
    return model, tokenizer

@torch.no_grad()
def run_gpt2_model(model: GPT2LMHeadModel,
                   tokenizer: GPT2Tokenizer,
                   dataset,
                   output: str,
                   batch_size: int = 8,
                   max_length: int = 128) -> None:
    device = model.device
    criterion = nn.NLLLoss(reduction='none')
    log_softmax = nn.LogSoftmax(dim=1)

    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        labels = [item['target'] for item in batch]
        languages = [item['language'] for item in batch]
        sources = [item['source'] for item in batch]
        encoded = tokenizer(texts,
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            return_tensors='pt')
        encoded['labels'] = encoded['input_ids'].clone()
        return encoded, labels, languages, sources

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    with open(output, 'w') as fw:
        for encoded_input, labels, languages, sources in tqdm(dataloader):
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            input_ids = encoded_input['input_ids']
            target = encoded_input['labels']

            try:
                output = model(**encoded_input)
            except Exception:
                print('input_ids:', input_ids)
                raise

            logits = output.logits.to(device)  # (B, L, V)
            logits = rearrange(logits, 'B L V -> B V L')
            shift_logits = logits[..., :-1]
            shift_target = target[..., 1:]

            nll_loss = criterion(log_softmax(shift_logits), shift_target).squeeze()
            # Handle both batch and single-sequence case
            if nll_loss.ndim == 1:
                nll_loss = nll_loss.unsqueeze(0)

            for loss_seq, label, language, source in zip(nll_loss, labels, languages, sources):
                res = loss_seq.tolist()
                if not isinstance(res, list):
                    res = [res]
                try:
                    res_str = ' '.join(f'{num:.4f}' for num in res)
                except Exception:
                    print('input_ids:', input_ids)
                    print('logits.shape:', logits.shape)
                    print('res:', res)
                    raise
                else:
                    fw.write(f'{res_str}\t{label}\t{language}\t{source}\n')
                    
if __name__ == "__main__":
    args = get_args()
    MODEL = args.model
    MODEL_PATH = args.model_path
    MAX_LENGTH = args.max_length
    BATCH_SIZE = args.batch_size
    DEVELOPMENT = args.development
    
    train_df = json_to_dataframe(DATA_TRAIN_JSON)
    test_df = json_to_dataframe(DATA_TEST_JSON)
    if DEVELOPMENT:
        train_df = train_df.sample(frac=0.01, random_state=42)
        test_df = test_df.sample(frac=0.01, random_state=42)
    
    out_dir = os.path.join(RESULTS_DIR, "fourier_gpt", "likelihood_scores")
    os.makedirs(out_dir, exist_ok=True)
    
    model, tokenizer = load_model(MODEL_PATH, MODEL)
    run_gpt2_model(model, tokenizer, train_df, os.path.join(out_dir, "nll_train"), BATCH_SIZE, MAX_LENGTH)
    run_gpt2_model(model, tokenizer, test_df, os.path.join(out_dir, "nll_test"), BATCH_SIZE, MAX_LENGTH)