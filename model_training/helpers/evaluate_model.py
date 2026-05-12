import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import redis
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse

'''
python3 helpers/evaluate_model.py --model_path logs/rnn/2026-04-02_17-52-59_035938_rnn_pid238051/checkpoints --data_dir ../data/hdf5_data_final --eval_type val --csv_path ../data/t15_copyTaskData_description.csv --gpu_number 0

python3 helpers/evaluate_model.py --model_path logs/snn/2026-04-06/2026-04-06_12-20-33_991769_snn_pid695282/checkpoints --data_dir ../data/hdf5_data_final --eval_type val --csv_path ../data/t15_copyTaskData_description.csv --gpu_number 0
'''

MODEL_TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAINING_ROOT))

from model_training.models.gru_model import GRUDecoder
from model_training.models.snn_backbone import SpikingModel
from helpers.evaluate_model_helpers import *

_BERT_SCORE_FN = None
_BERT_SCORE_AVAILABLE = None


def safe_bert_score(reference_text, hypothesis_text):
    global _BERT_SCORE_FN, _BERT_SCORE_AVAILABLE

    if _BERT_SCORE_AVAILABLE is None:
        try:
            from bert_score import score as bert_score_fn
            _BERT_SCORE_FN = bert_score_fn
            _BERT_SCORE_AVAILABLE = True
        except Exception:
            _BERT_SCORE_AVAILABLE = False

    if _BERT_SCORE_AVAILABLE:
        precision, recall, f1 = _BERT_SCORE_FN(
            [hypothesis_text],
            [reference_text],
            model_type="bert-base-uncased",
            lang="en",
            rescale_with_baseline=False,
            device=str(device),
        )
        return float(f1.mean().item()), "bert_score"

    ref_tokens = reference_text.split()
    hyp_tokens = hypothesis_text.split()
    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0, "lexical_fallback"

    matches = sum(1 for t in hyp_tokens if t in ref_tokens)
    precision = matches / len(hyp_tokens)
    recall = matches / len(ref_tokens)
    if precision + recall == 0:
        return 0.0, "lexical_fallback"
    return 2 * precision * recall / (precision + recall), "lexical_fallback"

# argument parser for command line arguments
parser = argparse.ArgumentParser(description='Evaluate a pretrained RNN model on the copy task dataset.')
parser.add_argument('--model_path', type=str, default=None,
                    help='Path to the pretrained model directory (relative to the current working directory).')
parser.add_argument('--data_dir', type=str, default='../data/hdf5_data_final',
                    help='Path to the dataset directory (relative to the current working directory).')
parser.add_argument('--eval_type', type=str, default='val', choices=['val', 'test'],
                    help='Evaluation type: "val" for validation set, "test" for test set. '
                         'If "test", ground truth is not available.')
parser.add_argument('--csv_path', type=str, default='../data/t15_copyTaskData_description.csv',
                    help='Path to the CSV file with metadata about the dataset (relative to the current working directory).')
parser.add_argument('--gpu_number', type=int, default=1,
                    help='GPU number to use for RNN model inference. Set to -1 to use CPU.')
args = parser.parse_args()

# paths to model and data directories
# Note: these paths are relative to the current working directory
model_path = args.model_path
data_dir = args.data_dir

# define evaluation type
eval_type = args.eval_type  # can be 'val' or 'test'. if 'test', ground truth is not available

# load csv file
b2txt_csv_df = pd.read_csv(args.csv_path)

# load model args
model_args = OmegaConf.load(os.path.join(model_path, 'args.yaml'))

# set up gpu device
gpu_number = 0
if torch.cuda.is_available() and gpu_number >= 0:
    if gpu_number >= torch.cuda.device_count():
        raise ValueError(f'GPU number {gpu_number} is out of range. Available GPUs: {torch.cuda.device_count()}')
    device = f'cuda:{gpu_number}'
    device = torch.device(device)
    print(f'Using {device} for model inference.')
else:
    if gpu_number >= 0:
        print(f'GPU number {gpu_number} requested but not available.')
    print('Using CPU for model inference.')
    device = torch.device('cpu')

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

if model_args["model_type"] == "rnn":
    model = GRUDecoder(
        neural_dim = model_args['model']['n_input_features'],
        n_units = model_args['model']['n_units'], 
        n_days = len(model_args['dataset']['sessions']),
        n_classes = model_args['dataset']['n_classes'],
        rnn_dropout = model_args['model']['rnn_dropout'],
        input_dropout = model_args['model']['input_network']['input_layer_dropout'],
        n_layers = model_args['model']['n_layers'],
        patch_size = model_args['model']['patch_size'],
        patch_stride = model_args['model']['patch_stride'],
    )
elif model_args["model_type"] == "snn":
    model = SpikingModel(
        cfg = model_args["snn_config"],
        neural_dim = model_args["model"]["n_input_features"],
        n_days = len(model_args["dataset"]["sessions"]),
        n_classes = model_args["dataset"]["n_classes"],
        input_dropout = model_args["model"]["input_network"]["input_layer_dropout"],
        patch_size = model_args["model"]["patch_size"],
        patch_stride = model_args["model"]["patch_stride"],
        debug = False,
        day_adaptation_mode = model_args.get("day_adaptation_mode", "matrix"),
    )
else:
    raise ValueError(f"Unknown model_type: {model_args['model_type']}")

# load model weights
print(f'Looking for model checkpoint in {model_path}...')
checkpoint_path = os.path.join(model_path, 'best_checkpoint')
if not os.path.exists(checkpoint_path):
    checkpoint_path = os.path.join(model_path, 'best_checkpoint')

checkpoint = torch.load(checkpoint_path, weights_only=False)

# rename keys to not start with "module." or "_orig_mod."
clean_state_dict = {}
for key, value in checkpoint['model_state_dict'].items():
    clean_key = key.replace("module.", "").replace("_orig_mod.", "")
    clean_state_dict[clean_key] = value
checkpoint['model_state_dict'] = clean_state_dict
model.load_state_dict(checkpoint['model_state_dict'])  

# add model to device
model.to(device) 

# set model to eval mode
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

day_params = 0
for name, param in model.named_parameters():
    if 'day' in name.lower():  # lowercase to be safe
        day_params += param.numel()

print(f"Model has {day_params:,} day-specific parameters | "
      f"{(day_params / total_params) * 100:.2f}% of total parameters")

dtypes = {p.dtype for p in model.parameters()}
print(f"Parameter dtype(s): {dtypes}")

print(f"Current allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# load data for each session
test_data = {}
total_test_trials = 0
for session in model_args['dataset']['sessions']:
    files = [f for f in os.listdir(os.path.join(data_dir, session)) if f.endswith('.hdf5')]
    if f'data_{eval_type}.hdf5' in files:
        eval_file = os.path.join(data_dir, session, f'data_{eval_type}.hdf5')

        data = load_h5py_file(eval_file, b2txt_csv_df)
        test_data[session] = data

        total_test_trials += len(test_data[session]["neural_features"])
        print(f'Loaded {len(test_data[session]["neural_features"])} {eval_type} trials for session {session}.')
        break
print(f'Total number of {eval_type} trials: {total_test_trials}')
print()


# put neural data through the pretrained model to get phoneme predictions (logits)
with tqdm(total=total_test_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
    for session, data in test_data.items():

        data['logits'] = []
        data['pred_seq'] = []
        input_layer = model_args['dataset']['sessions'].index(session)
        
        for trial in range(len(data['neural_features'])):
            # get neural input for the trial
            neural_input = data['neural_features'][trial]

            # add batch dimension
            neural_input = np.expand_dims(neural_input, axis=0)

            # convert to torch tensor
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.float32)

            # run decoding step
            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
            data['logits'].append(logits)

            pbar.update(1)
pbar.close()

# we probably have to base it on insertions/deletions
def phoneme_error_rate(true_phonemes, predicted_phonemes):
    # Remove word boundaries
    #true = [p for p in true_phonemes if p != ' | ']
    #pred = [p for p in predicted_phonemes if p != ' | ']
    true = true_phonemes
    pred = predicted_phonemes


    # Initialize distance matrix
    m, n = len(true), len(pred)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Compute edit distance
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if true[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # deletion
                    dp[i][j - 1],    # insertion
                    dp[i - 1][j - 1] # substitution
                )

    edit_distance = dp[m][n]
    per = edit_distance / m if m > 0 else 0.0
    return per, edit_distance


predictions = []
true_seq_length = 0
edit_distance_total = 0
bert_scores = []
bert_score_mode = None
# convert logits to phoneme sequences and print them out
for session, data in test_data.items():
    data['pred_seq'] = []
    print(data.keys())
    print(data['sentence_label'])
    for trial in range(len(data['logits'])):
        logits = data['logits'][trial][0]
        pred_seq = np.argmax(logits, axis=-1)
        # remove blanks (0)
        pred_seq = [int(p) for p in pred_seq if p != 0]
        # remove consecutive duplicates
        pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]
        # convert to phonemes
        pred_seq_phoneme = [LOGIT_TO_PHONEME[p] for p in pred_seq]
        # add to data
        data['pred_seq'].append(pred_seq)

        # print out the predicted sequences
        block_num = data['block_num'][trial]
        trial_num = data['trial_num'][trial]

        row = {
            'session': session,
            'block_num': block_num,
            'trial_num': trial_num,
            'predicted_logits' : pred_seq
            #'predicted_sequence': ' '.join(pred_seq),
        }

        print(f'Session: {session}, Block: {block_num}, Trial: {trial_num}')
        if eval_type == 'val':
            sentence_label = data['sentence_label'][trial]
            true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
            true_seq_phoneme = [LOGIT_TO_PHONEME[p] for p in true_seq]
            true_seq_text = ' '.join(true_seq_phoneme)
            pred_seq_text = ' '.join(pred_seq_phoneme)
            bert_score_value, bert_score_mode = safe_bert_score(true_seq_text, pred_seq_text)
            bert_scores.append(bert_score_value)

            row['sentence_label'] = sentence_label
            row['true_sequence'] = true_seq_text
            row['bert_score'] = bert_score_value

            print(f'Sentence label:      {sentence_label}')
            print(f'True sequence:       {true_seq_text}')
            print(f'BERT score:          {bert_score_value:.5f}')

        print(f'Predicted Sequence:  {" ".join(pred_seq_phoneme)}')
        #print(true_seq_phoneme)
        #print(pred_seq_phoneme)
        predictions.append(row)
        if eval_type == 'val':
            per, edit_distance = phoneme_error_rate(true_seq_phoneme, pred_seq_phoneme)
            #print(edit_distance)
            #print(len(true_seq_phoneme))
            true_seq_length += len(true_seq_phoneme)
            edit_distance_total += edit_distance
            #print(per.round(3))
            print()

if eval_type == 'val':
    per_final = edit_distance_total / true_seq_length
    print(f'Total edit distance: {edit_distance_total}')
    print(f'Total true sequence length: {true_seq_length}')
    print(f'Aggregate Phoneme Error Rate (PER): {per_final:.5f}')
if eval_type == 'val':
    mean_bert_score = float(np.mean(bert_scores)) if len(bert_scores) > 0 else float('nan')
    print(f'Mean semantic BERT score ({bert_score_mode}): {mean_bert_score:.5f}')
#df = pd.DataFrame(predictions)
#csv_filename = f'predictions_{eval_type}_2.csv'
#df.to_csv(csv_filename, index=False)

# language model inference via redis
# make sure that the standalone language model is running on the localhost redis ip
# see README.md for instructions on how to run the language model
r = redis.Redis(host='localhost', port=6379, db=0)
r.flushall()  # clear all streams in redis

# define redis streams for the remote language model
remote_lm_input_stream = 'remote_lm_input'
remote_lm_output_partial_stream = 'remote_lm_output_partial'
remote_lm_output_final_stream = 'remote_lm_output_final'

# set timestamps for last entries seen in the redis streams
remote_lm_output_partial_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_output_final_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_resetting_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_finalizing_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_updating_lastEntrySeen = get_current_redis_time_ms(r)

lm_results = {
    'session': [],
    'block': [],
    'trial': [],
    'true_sentence': [],
    'pred_sentence': [],
}

# loop through all trials and put logits into the remote language model to get text predictions
# note: this takes ~15-20 minutes to run on the entire test split with the 5-gram LM + OPT rescoring (RTX 4090)
with tqdm(total=total_test_trials, desc='Running remote language model', unit='trial') as pbar:
    for session in test_data.keys():
        for trial in range(len(test_data[session]['logits'])):
            # get trial logits and rearrange them for the LM
            logits = rearrange_speech_logits_pt(test_data[session]['logits'][trial])[0]

            # reset language model
            remote_lm_done_resetting_lastEntrySeen = reset_remote_language_model(r, remote_lm_done_resetting_lastEntrySeen)
            
            
            # update language model parameters
            #remote_lm_done_updating_lastEntrySeen = update_remote_lm_params(
            #    r,
            #    remote_lm_done_updating_lastEntrySeen,
            #    acoustic_scale=0.35,
            #    blank_penalty=90.0,
            #    alpha=0.55,
            #)


            # put logits into LM
            remote_lm_output_partial_lastEntrySeen, decoded = send_logits_to_remote_lm(
                r,
                remote_lm_input_stream,
                remote_lm_output_partial_stream,
                remote_lm_output_partial_lastEntrySeen,
                logits,
            )

            # finalize remote LM
            remote_lm_output_final_lastEntrySeen, lm_out = finalize_remote_lm(
                r,
                remote_lm_output_final_stream,
                remote_lm_output_final_lastEntrySeen,
            )

            # get the best candidate sentence
            best_candidate_sentence = lm_out['candidate_sentences'][0]

            # store results
            lm_results['session'].append(session)
            lm_results['block'].append(test_data[session]['block_num'][trial])
            lm_results['trial'].append(test_data[session]['trial_num'][trial])
            if eval_type == 'val':
                lm_results['true_sentence'].append(test_data[session]['sentence_label'][trial])
            else:
                lm_results['true_sentence'].append(None)
            lm_results['pred_sentence'].append(best_candidate_sentence)

            # update progress bar
            pbar.update(1)
pbar.close()


# if using the validation set, lets calculate the aggregate word error rate (WER)
if eval_type == 'val':
    total_true_length = 0
    total_edit_distance = 0

    lm_results['edit_distance'] = []
    lm_results['num_words'] = []

    for i in range(len(lm_results['pred_sentence'])):
        true_sentence = remove_punctuation(lm_results['true_sentence'][i]).strip()
        pred_sentence = remove_punctuation(lm_results['pred_sentence'][i]).strip()
        ed = editdistance.eval(true_sentence.split(), pred_sentence.split())

        total_true_length += len(true_sentence.split())
        total_edit_distance += ed

        lm_results['edit_distance'].append(ed)
        lm_results['num_words'].append(len(true_sentence.split()))

        print(f'{lm_results["session"][i]} - Block {lm_results["block"][i]}, Trial {lm_results["trial"][i]}')
        print(f'True sentence:       {true_sentence}')
        print(f'Predicted sentence:  {pred_sentence}')
        print(f'WER: {ed} / {100 * len(true_sentence.split())} = {ed / len(true_sentence.split()):.2f}%')
        print()

    print(f'Total true sentence length: {total_true_length}')
    print(f'Total edit distance: {total_edit_distance}')
    print(f'Aggregate Word Error Rate (WER): {100 * total_edit_distance / total_true_length:.2f}%')


# write predicted sentences to a csv file. put a timestamp in the filename (YYYYMMDD_HHMMSS)
output_file = os.path.join(model_path, f'baseline_rnn_{eval_type}_predicted_sentences_{time.strftime("%Y%m%d_%H%M%S")}.csv')
ids = [i for i in range(len(lm_results['pred_sentence']))]
df_out = pd.DataFrame({'id': ids, 'text': lm_results['pred_sentence']})
df_out.to_csv(output_file, index=False)
