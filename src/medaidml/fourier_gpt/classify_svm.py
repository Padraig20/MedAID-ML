import os
import pandas as pd
import numpy as np

from typing import Tuple, Union

from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from medaidml import RESULTS_DIR
from medaidml.utils import split_val_test

# Extract features by linear interpolation
def get_features(spectrum_data: Union[str, pd.DataFrame],
                 interp_len: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if isinstance(spectrum_data, str):
        df = pd.read_csv(spectrum_data)
    else:
        df = spectrum_data

    # If `sid` column does not exist, create it
    if 'sid' not in df.columns:
        df['sdiff']  = df['freq'] < df['freq'].shift(1, fill_value=0)
        df['sdiff'] = df['sdiff'].astype(int)
        df['sid'] = df['sdiff'].cumsum()

    features_interp = []
    labels = []
    languages = []
    sources = []
    for _, group in df.groupby('sid'):
        freqs = group['freq'].values
        features = group['power'].values
        new_freq = np.linspace(0, 0.5, interp_len)
        new_feat = np.interp(new_freq, freqs, features)
        features_interp.append(new_feat)
        labels.append(group['label'].values[0])
        languages.append(group['language'].values[0])
        sources.append(group['source'].values[0])

    return np.array(features_interp), np.array(labels), np.array(languages), np.array(sources)

def fit_svm(fft_data: pd.DataFrame) -> Pipeline:
    print("Getting features...")
    x, y, _, _ = get_features(fft_data)

    cls = make_pipeline(StandardScaler(),
                        SelectKBest(k=120),
                        SVC(gamma='auto', kernel='rbf', C=1))
    
    print("Fitting pipeline...")
    cls = cls.fit(x, y)

    return cls

def evaluate_svm(fft_data: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    print("Getting features...")
    x, y, languages, sources = get_features(fft_data)

    print("Evaluating pipeline...")
    y_pred = model.predict(x)
    
    df = pd.DataFrame({
        'Ground Truth': y,
        'Prediction': y_pred,
        'language': languages,
        'source': sources
    })
    
    return df

if __name__ == "__main__":
    fft_dir = os.path.join(RESULTS_DIR, "fourier_gpt")
    in_dir = os.path.join(fft_dir, "fft_transformed")
    
    train_file = os.path.join(in_dir, "train.csv")
    test_file = os.path.join(in_dir, "test.csv")
    
    no_dataleak_df = pd.read_csv(test_file)
    all_train_df = pd.read_csv(train_file)
    
    for i in range(1, 6):
        train_df, val_df, test_df = split_val_test(all_train_df, no_dataleak_df, seed=i) 
           
        model = fit_svm(train_df)
        results_val = evaluate_svm(val_df, model)
        results_test = evaluate_svm(test_df, model)
        results_no_dataleak = evaluate_svm(no_dataleak_df, model)
        
        print(f"Results for seed {i}:")
        print(f"Validation Accuracy: {accuracy_score(results_val['Ground Truth'], results_val['Prediction'])}")
        print(f"Test Accuracy: {accuracy_score(results_test['Ground Truth'], results_test['Prediction'])}")
        print(f"No Data Leak Accuracy: {accuracy_score(results_no_dataleak['Ground Truth'], results_no_dataleak['Prediction'])}")
        print("-" * 30)
        
        os.makedirs(os.path.join(fft_dir, i), exist_ok=True)
        results_test.to_csv(os.path.join(fft_dir, i, "results_test.csv"), index=False)
        results_no_dataleak.to_csv(os.path.join(fft_dir, i, "results_no_dataleak.csv"), index=False)

    print("SVM classification completed. Results saved to respective directories.")
