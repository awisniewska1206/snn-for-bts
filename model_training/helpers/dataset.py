import os
from pathlib import Path
import torch
from torch.utils.data import Dataset 
import h5py
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import math 
import csv
from collections import defaultdict

class BrainToTextDataset(Dataset):
    '''
    Dataset for brain-to-text data
    
    Returns an entire batch of data instead of a single example
    '''

    def __init__(
            self, 
            trial_indicies,
            n_batches,
            split = 'train', 
            batch_size = 64, 
            days_per_batch = 1, 
            random_seed = -1,
            must_include_days = None,
            feature_subset = None
            ): 
        '''
        trial_indicies:  (dict)      - dictionary with day numbers as keys and lists of trial indices as values
        n_batches:       (int)       - number of random training batches to create
        split:           (string)    - string specifying if this is a train or test dataset
        batch_size:      (int)       - number of examples to include in batch returned from __getitem_()
        days_per_batch:  (int)       - how many unique days can exist in a batch; this is important for making sure that updates 
                                       to individual day layers in the GRU are not excesively noisy. Validation data will always have 1 day per batch
        random_seed:     (int)       - seed to set for randomly assigning trials to a batch. If set to -1, trial assignment will be random
        must_include_days ([int])    - list of days that must be included in every batch
        feature_subset  ([int])      - list of neural feature indicies that should be the only features included in the neural data 
         '''
        
        # Set random seed for reproducibility
        if random_seed is not None and int(random_seed) >= 0:
            self.rng = np.random.default_rng(int(random_seed))
        else:
            self.rng = np.random.default_rng()

        self.split = split

        # Ensure the split is valid
        if self.split not in ['train', 'test']:
            raise ValueError(f'split must be either "train" or "test". Received {self.split}')
        
        self.days_per_batch = days_per_batch

        self.batch_size = batch_size

        self.n_batches = n_batches

        self.days = {}
        self.n_trials = 0 
        self.trial_indicies = trial_indicies
        self.n_days = len(trial_indicies.keys())
        self.trial_refs = {}

        self.feature_subset = feature_subset

        # Calculate total number of trials in the dataset
        for d in trial_indicies:
            self.trial_refs[d] = self._normalize_trial_refs(trial_indicies[d])
            self.n_trials += len(self.trial_refs[d])

        self.sampleable_train_days = [
            d for d in self.trial_indicies.keys() if len(self.trial_refs[d]) > 0
        ]

        if must_include_days is not None and len(must_include_days) > days_per_batch:
            raise ValueError(f'must_include_days must be less than or equal to days_per_batch. Received {must_include_days} and days_per_batch {days_per_batch}')
        
        if must_include_days is not None and len(must_include_days) > self.n_days and split != 'train':
            raise ValueError(f'must_include_days is not valid for test data. Received {must_include_days} and but only {self.n_days} in the dataset')
        
        if must_include_days is not None:
            # Map must_include_days to correct indicies if they are negative
            for i, d in enumerate(must_include_days):
                if d < 0: 
                    must_include_days[i] = self.n_days + d

        self.must_include_days = must_include_days    

        # Ensure that the days_per_batch is not greater than the number of days in the dataset. Raise error
        if self.split == 'train' and len(self.sampleable_train_days) == 0:
            raise ValueError("No training trials are available for any day in this dataset.")

        if self.split == 'train' and self.days_per_batch > len(self.sampleable_train_days):
            raise ValueError(
                f"Requested days_per_batch: {days_per_batch} is greater than available non-empty training days "
                f"{len(self.sampleable_train_days)}."
            )
           
        
        if self.split == 'train':
            self.batch_index = self.create_batch_index_train()
        else: 
            self.batch_index = self.create_batch_index_test()
            self.n_batches = len(self.batch_index.keys()) # The validation data has a fixed amount of data 
    
    def __len__(self):
        ''' 
        How many batches are in this dataset. 
        Because training data is sampled randomly, there is no fixed dataset length, 
        however this method is required for DataLoader to work 
        '''
        return self.n_batches
    
    def __getitem__(self, idx):
        ''' 
        Gets an entire batch of data from the dataset, not just a single item
        '''
        batch = {
            'input_features' : [],
            'seq_class_ids' : [],
            'n_time_steps' : [],
            'phone_seq_lens' : [],
            'day_indicies' : [],
            'session_ids' : [],
            'transcriptions' : [],
            'block_nums' : [],
            'trial_nums' : [],
        }

        index = self.batch_index[idx]

        # Iterate through each day in the index
        for d in index.keys():
            refs_by_session = defaultdict(list)
            for ref in index[d]:
                refs_by_session[ref['session_path']].append(ref)

            for session_path, refs in refs_by_session.items():
                with h5py.File(session_path, 'r') as f:
                    for ref in refs:
                        try: 
                            t = int(ref['trial_idx'])
                            g = f[f'trial_{t:04d}']

                            input_features = torch.from_numpy(g['input_features'][:]) # neural data
                            if self.feature_subset:
                                input_features = input_features[:,self.feature_subset]

                            batch['input_features'].append(input_features)

                            batch['seq_class_ids'].append(torch.from_numpy(g['seq_class_ids'][:]))  # phoneme labels
                            batch['transcriptions'].append(torch.from_numpy(g['transcription'][:])) # character level transcriptions
                            batch['n_time_steps'].append(g.attrs['n_time_steps']) # number of time steps in the trial - required since we are padding
                            batch['phone_seq_lens'].append(g.attrs['seq_len']) # number of phonemes in the label - required since we are padding
                            batch['day_indicies'].append(int(d)) # day index of each trial - required for the day specific layers 
                            batch['session_ids'].append(int(d)) # explicit session id alias for session-adaptive components
                            batch['block_nums'].append(g.attrs['block_num'])
                            batch['trial_nums'].append(g.attrs['trial_num'])
                        
                        except Exception as e:
                            print(f'Error loading trial {ref} from session {session_path}: {e}')
                            continue

        # Pad data to form a cohesive batch
        batch['input_features'] = pad_sequence(batch['input_features'], batch_first = True, padding_value = 0)
        batch['seq_class_ids'] = pad_sequence(batch['seq_class_ids'], batch_first = True, padding_value = 0)

        batch['n_time_steps'] = torch.tensor(batch['n_time_steps']) 
        batch['phone_seq_lens'] = torch.tensor(batch['phone_seq_lens'])
        batch['day_indicies'] = torch.tensor(batch['day_indicies'])
        batch['session_ids'] = torch.tensor(batch['session_ids'])
        batch['transcriptions'] = torch.stack(batch['transcriptions'])
        batch['block_nums'] = torch.tensor(batch['block_nums'])
        batch['trial_nums'] = torch.tensor(batch['trial_nums'])

        return batch

    def _normalize_trial_refs(self, day_payload):
        if 'trial_refs' in day_payload:
            return [
                {
                    'trial_idx': int(ref['trial_idx']),
                    'session_path': str(ref['session_path']),
                }
                for ref in day_payload['trial_refs']
            ]

        if 'trials' not in day_payload or 'session_path' not in day_payload:
            raise ValueError(
                "Each day payload must contain either {'trials', 'session_path'} or {'trial_refs'}."
            )

        return [
            {
                'trial_idx': int(trial_idx),
                'session_path': str(day_payload['session_path']),
            }
            for trial_idx in day_payload['trials']
        ]
    

    def create_batch_index_train(self):
        '''
        Create an index that maps a batch_number to batch_size number of trials

        Each batch will have days_per_batch unique days of data, with the number of trials for each day evenly split between the days 
        (or as even as possible if batch_size is not divisible by days_per_batch)
        '''

        batch_index = {}
        sampleable_days = list(self.sampleable_train_days)

        if self.must_include_days is not None:
            empty_must_include_days = [d for d in self.must_include_days if len(self.trial_refs.get(d, [])) == 0]
            if empty_must_include_days:
                raise ValueError(
                    f"must_include_days contains day indices with no training trials: {empty_must_include_days}"
                )

        # Precompute the days that are not in must_include_days
        if self.must_include_days is not None:
            non_must_include_days = [d for d in sampleable_days if d not in self.must_include_days]

        for batch_idx in range(self.n_batches):
            batch = {}

            # Which days will be used for this batch. Picked randomly without replacement
            # TODO: In the future we may want to consider sampling days in proportion to the number of trials in each day 

            # If must_include_days is not empty, we will use those days and then randomly sample the rest
            if self.must_include_days is not None and len(self.must_include_days) > 0:

                days = np.concatenate(
                    (
                        self.must_include_days,
                        self.rng.choice(
                            non_must_include_days,
                            size=self.days_per_batch - len(self.must_include_days),
                            replace=False,
                        ),
                    )
                )
            
            # Otherwise we will select random days without replacement
            else: 
                days = self.rng.choice(sampleable_days, size=self.days_per_batch, replace=False)
            
            # How many trials will be sampled from each day
            num_trials = math.ceil(self.batch_size / self.days_per_batch) # Use ceiling to make sure we get at least batch_size trials

            for d in days:

                # Trials are sampled with replacement, so if a day has less than (self.batch_size / days_per_batch trials) trials, it won't be a problem
                ref_indices = self.rng.choice(len(self.trial_refs[d]), size=num_trials, replace=True)
                batch[d] = [self.trial_refs[d][int(ref_idx)] for ref_idx in ref_indices]

            # Remove extra trials
            extra_trials = (num_trials * len(days)) - self.batch_size

            # While we still have extra trials, remove the last trial from a random day
            while extra_trials > 0: 
                d = self.rng.choice(days)
                batch[d] = batch[d][:-1]
                extra_trials -= 1

            batch_index[batch_idx] = batch

        return batch_index
    
    def create_batch_index_test(self):
        '''
        Create an index that is all validation/testing data in batches of up to self.batch_size

        If a day does not have at least self.batch_size trials, then the batch size will be less than self.batch_size

        This index will ensures that every trial in the validation set is seen once and only once
        '''
        batch_index = {}
        batch_idx = 0
        
        for d in self.trial_indicies.keys():

            # Calculate how many batches we need for this day
            num_trials = len(self.trial_refs[d])
            num_batches = (num_trials + self.batch_size - 1) // self.batch_size 
            
            # Create batches for this day
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_trials)
                
                # Get the trial indices for this batch
                batch_trials = self.trial_refs[d][start_idx:end_idx]
                
                # Add to batch_index
                batch_index[batch_idx] = {d : batch_trials}
                batch_idx += 1
        
        return batch_index
        
def train_test_split_indicies(file_paths, test_percentage = 0.1, seed = -1, bad_trials_dict = None):
    '''
    Split data from file_paths into train and test splits 
    Returns two dictionaries that detail which trials in each day will be a part of that split:
    Example: 
        {
            0: trials[1,2,3], session_path: 'path'
            1: trials[2,5,6], session_path: 'path'
        }

    Args:
        file_paths (list): List of file paths to the hdf5 files containing the data
        test_percentage (float): Percentage of trials to use for testing. 0 will use all trials for training, 1 will use all trials for testing
        seed (int): Seed for reproducibility. If set to -1, the split will be random
        bad_trials_dict (dict): Dictionary of trials to exclude from the dataset. Formatted as:
            {
                'session_name_1': {block_num_1: [trial_nums], block_num_2: [trial_nums], ...},
                'session_name_2': {block_num_1: [trial_nums], block_num_2: [trial_nums], ...},
                ...
            }
    '''
    if seed is not None and int(seed) >= 0:
        rng = np.random.default_rng(int(seed))
    else:
        rng = np.random.default_rng()

    # Get trials in each day
    trials_per_day = {}
    for i, path in enumerate(file_paths):
        session = Path(path).parent.name

        good_trial_indices = []

        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                num_trials = len(list(f.keys()))
                for t in range(num_trials):
                    key = f'trial_{t:04d}'
                    
                    block_num = f[key].attrs['block_num']
                    trial_num = f[key].attrs['trial_num']

                    if (
                        bad_trials_dict is not None
                        and session in bad_trials_dict
                        and str(block_num) in bad_trials_dict[session]
                        and trial_num in bad_trials_dict[session][str(block_num)]
                    ):
                        # print(f'Bad trial: {session}_{block_num}_{trial_num}')
                        continue

                    good_trial_indices.append(t)

        trials_per_day[i] = {'num_trials': len(good_trial_indices), 'trial_indices': good_trial_indices, 'session_path': path}

    # Pick test_percentage of trials from each day for testing and (1 - test_percentage) for training
    train_trials = {}
    test_trials = {}

    for day in trials_per_day.keys():

        num_trials = trials_per_day[day]['num_trials']

        # Generate all trial indices for this day (assuming 0-indexed)
        all_trial_indices = trials_per_day[day]['trial_indices']

        # If test_percentage is 0 or 1, we can just assign all trials to either train or test
        if test_percentage == 0:
            train_trials[day] = {'trials' : all_trial_indices, 'session_path' : trials_per_day[day]['session_path']}
            test_trials[day] = {'trials' : [], 'session_path' : trials_per_day[day]['session_path']}
            continue
        
        elif test_percentage == 1:
            train_trials[day] = {'trials' : [], 'session_path' : trials_per_day[day]['session_path']}
            test_trials[day] = {'trials' : all_trial_indices, 'session_path' : trials_per_day[day]['session_path']}
            continue    

        else:
            # Calculate how many trials to use for testing
            num_test = max(1, int(num_trials * test_percentage))
            
            # Randomly select indices for testing
            test_indices = rng.choice(all_trial_indices, size=num_test, replace=False).tolist()
            
            # Remaining indices go to training
            train_indices = [idx for idx in all_trial_indices if idx not in test_indices]
            
            # Store the split indices
            train_trials[day] = {'trials' : train_indices, 'session_path' : trials_per_day[day]['session_path']}
            test_trials[day] = {'trials' : test_indices, 'session_path' : trials_per_day[day]['session_path']}
    
    return train_trials, test_trials


def _session_name_to_date(session_name):
    parts = session_name.split(".")
    if len(parts) < 4:
        raise ValueError(f"Session name does not match expected pattern: {session_name}")
    return f"{parts[1]}-{parts[2]}-{parts[3]}"


def _discover_session_trial_refs(dataset_dir, session_name, source_filenames):
    session_dir = Path(dataset_dir) / session_name
    block_trials = defaultdict(list)

    for source_name in source_filenames:
        source_path = session_dir / source_name
        if not source_path.exists():
            continue

        with h5py.File(source_path, "r") as handle:
            for trial_key in handle.keys():
                trial_idx = int(trial_key.split("_")[-1])
                attrs = handle[trial_key].attrs
                block_num = int(attrs["block_num"])
                trial_num = int(attrs["trial_num"])
                block_trials[block_num].append(
                    {
                        "trial_idx": trial_idx,
                        "trial_num": trial_num,
                        "session_path": str(source_path),
                    }
                )

    for block_num in block_trials:
        block_trials[block_num].sort(
            key=lambda item: (item["trial_num"], item["trial_idx"], item["session_path"])
        )

    return block_trials


def load_trial_indices_from_split_csv(
    split_csv_path,
    dataset_dir,
    sessions,
    source_filenames=None,
):
    if source_filenames is None:
        source_filenames = ["data_train.hdf5", "data_val.hdf5", "data_test.hdf5"]

    split_csv_path = Path(split_csv_path)
    dataset_dir = Path(dataset_dir)

    rows_by_date = defaultdict(lambda: defaultdict(list))
    with split_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "split", "block_number", "sentence_in_block"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Split CSV is missing required columns: {sorted(missing)}"
            )

        for row_idx, row in enumerate(reader):
            enriched_row = dict(row)
            enriched_row["_row_idx"] = int(row_idx)
            rows_by_date[row["date"]][row["split"]].append(enriched_row)

    split_payloads = {"train": {}, "val": {}, "test": {}}
    session_date_map = {session: _session_name_to_date(session) for session in sessions}

    for day_idx, session_name in enumerate(sessions):
        date = session_date_map[session_name]
        block_trials = _discover_session_trial_refs(
            dataset_dir=dataset_dir,
            session_name=session_name,
            source_filenames=source_filenames,
        )
        used_keys = set()
        split_rows = rows_by_date.get(date, {})
        rows_by_block = defaultdict(list)
        for split_name in ("train", "val", "test"):
            for row in split_rows.get(split_name, []):
                rows_by_block[int(row["block_number"])].append(row)

        row_to_trial_ref = {}
        for block_number, requested_rows in rows_by_block.items():
            ordered_requested_rows = sorted(
                requested_rows,
                key=lambda row: (
                    int(row["sentence_in_block"]),
                    int(row.get("sentence_index_in_day", 0)),
                    int(row["_row_idx"]),
                ),
            )
            block_records = block_trials.get(block_number, [])
            available_count = len(block_records)
            requested_count = len(ordered_requested_rows)
            matched_count = min(requested_count, available_count)

            if requested_count != available_count:
                print(
                    f"[split-loader] session={session_name} block={block_number}: "
                    f"requested {requested_count} sentence slots from CSV but found "
                    f"{available_count} trial(s) in HDF5. Using first {matched_count} by rank order."
                )

            for row, chosen in zip(ordered_requested_rows[:matched_count], block_records[:matched_count]):
                unique_key = (chosen["session_path"], chosen["trial_idx"])
                if unique_key in used_keys:
                    raise ValueError(
                        f"Duplicate trial assignment detected for session={session_name}, "
                        f"block={block_number}, trial_idx={chosen['trial_idx']}."
                    )
                used_keys.add(unique_key)
                row_to_trial_ref[int(row["_row_idx"])] = {
                    "trial_idx": int(chosen["trial_idx"]),
                    "session_path": str(chosen["session_path"]),
                }

            if requested_count > available_count:
                dropped_rows = ordered_requested_rows[matched_count:]
                if dropped_rows:
                    dropped_desc = ", ".join(
                        str(int(row["sentence_in_block"])) for row in dropped_rows[:8]
                    )
                    suffix = "..." if len(dropped_rows) > 8 else ""
                    print(
                        f"[split-loader] dropped {len(dropped_rows)} row(s) for session={session_name}, "
                        f"block={block_number}; sentence_in_block values: {dropped_desc}{suffix}"
                    )

        for split_name in split_payloads.keys():
            day_rows = rows_by_date.get(date, {}).get(split_name, [])
            ordered_rows = sorted(
                day_rows,
                key=lambda row: int(row.get("sentence_index_in_day", 0)),
            )
            trial_refs = []
            for row in ordered_rows:
                chosen = row_to_trial_ref.get(int(row["_row_idx"]))
                if chosen is None:
                    continue
                trial_refs.append(chosen)

            split_payloads[split_name][day_idx] = {
                "trial_refs": trial_refs,
                "session_path": str((dataset_dir / session_name)),
            }

    return split_payloads["train"], split_payloads["val"], split_payloads["test"]
