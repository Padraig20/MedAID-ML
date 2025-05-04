from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch

from medaidml import DATA_TEST_JSON
from medaidml.utils import json_to_dataframe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Integrated Gradients for custom huggingface model")
    parser.add_argument("--model_name", type=str, default="mBERT-detect", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for results")
    parser.add_argument("--development", action="store_true", help="Use a smaller sample for development")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top tokens to display")
    parser.add_argument("--ngram", type=int, default=0, help="Use n-gram attribution (0 for none)")
    return parser.parse_args()

def load_model_and_tokenizer(model_name: str):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except ValueError:
        print(f"Model '{model_name}' not found online. Loading local model instead.")
        model = AutoModelForSequenceClassification.from_pretrained(f"{model_name}/model")
        tokenizer = AutoTokenizer.from_pretrained(f"{model_name}/tokenizer")
    model.to(device)
    return model, tokenizer

def compute_attributions(ig, model, tokenizer, text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    embeddings = model.get_input_embeddings()(input_ids)

    attributions, _ = ig.attribute(
        embeddings,
        additional_forward_args=(attention_mask,),
        return_convergence_delta=True
    )

    token_attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

    return tokens, token_attributions

def compute_ngram_attributions(tokens, token_attributions, n=2):
    ngram_attributions = []
    ngram_tokens = []

    for i in range(len(tokens) - n + 1):
        ngram = tokens[i:i + n]
        attribution = np.mean(token_attributions[i:i + n])
        ngram_tokens.append(" ".join(ngram))
        ngram_attributions.append(attribution)

    return ngram_tokens, ngram_attributions

def custom_forward(embeddings, attention_mask):
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    return outputs.logits[:, 1]  # explain class 1 (AI)

if __name__ == "__main__":
    args = parse_args()
    DEVELOPMENT = args.development
    MODEL_NAME = args.model_name
    OUTPUT_DIR = args.output_dir
    TOP_K = args.top_k
    NGRAM = args.ngram

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    model.eval()

    data = json_to_dataframe(DATA_TEST_JSON)
    global_token_scores = defaultdict(list)  # (token, language) -> list of scores

    if DEVELOPMENT:
        data = data.sample(frac=0.05, random_state=42)

    ig = IntegratedGradients(custom_forward)

    for _, instance in tqdm(data.iterrows(), desc="Computing attributions", total=len(data), unit="instance"):
        tokens, token_attributions = compute_attributions(ig, model, tokenizer, instance['text'])

        if NGRAM > 0:
            ngram_tokens, ngram_attributions = compute_ngram_attributions(tokens, token_attributions, NGRAM)
            for token, score in zip(ngram_tokens, ngram_attributions):
                global_token_scores[(token, instance['language'])].append(score)
        else:
            for token, score in zip(tokens, token_attributions):
                global_token_scores[(token, instance['language'])].append(score)

    with open("token_scores.csv", "w") as f:
        f.write("token,language,score\n")
        for (token, language), scores in global_token_scores.items():
            f.write(f"\"{token}\",{language},{np.mean(scores)}\n")

    for language in ['en', 'de', 'es', 'fr']:
        print(f"Top {TOP_K} tokens for language: {language}")
        language_token_scores = {
            token: np.mean(scores)
            for (token, lang), scores in global_token_scores.items()
            if lang == language
        }
        sorted_tokens = sorted(language_token_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        for token, score in sorted_tokens:
            print(f"{token:15s} -> {score:.4f}")
        print("-" * 30)
