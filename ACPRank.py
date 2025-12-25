
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from tqdm import tqdm
from datetime import datetime
import json
import gc
import pandas as pd
from sklearn.metrics import ndcg_score, average_precision_score
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score, average_precision_score, roc_auc_score
from src.models.layers import TransformerEncoderLayer
from src.models.peptide_mlm import PeptideMLMModel
from src.models.peptide_ranking import PeptideRankingModel


class CostSensitiveRankingLoss(tf.keras.losses.Loss):
    def __init__(self, threshold=4.3010, false_high_penalty=5.0, false_low_penalty=2.0,
                 rank_diff_factor=1.5, top_rank_focus=2.0, margin=0.1,
                 name="improved_cost_sensitive_ranking_loss"):
        super().__init__(name=name)
        self.threshold = threshold
        self.false_high_penalty = false_high_penalty
        self.false_low_penalty = false_low_penalty
        self.rank_diff_factor = rank_diff_factor
        self.top_rank_focus = top_rank_focus
        self.margin = margin

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1, 2])
        y_pred = tf.reshape(y_pred, [-1, 2])

        higher_true = y_true[:, 0]
        lower_true = y_true[:, 1]
        higher_pred = y_pred[:, 0]
        lower_pred = y_pred[:, 1]

        basic_loss = tf.maximum(0., self.margin - (higher_pred - lower_pred))

        higher_active = tf.cast(tf.greater(higher_true, self.threshold), tf.float32)
        lower_active = tf.cast(tf.greater(lower_true, self.threshold), tf.float32)

        severe_error = higher_active * (1.0 - lower_active)
        both_active = higher_active * lower_active

        batch_max = tf.reduce_max(y_true)
        batch_min = tf.reduce_min(y_true)
        range_value = batch_max - batch_min + 1e-6

        norm_higher = (higher_true - batch_min) / range_value
        norm_lower = (lower_true - batch_min) / range_value

        rank_diff = tf.abs(norm_higher - norm_lower)
        rank_diff_penalty = both_active * rank_diff * self.rank_diff_factor

        top_rank_weight = norm_higher * self.top_rank_focus
        active_top_penalty = higher_active * top_rank_weight

        penalty_factor = (1.0 +
                         severe_error * (self.false_high_penalty - 1.0) +
                         both_active * (self.false_low_penalty - 1.0) +
                         rank_diff_penalty +
                         active_top_penalty)

        weighted_loss = basic_loss * penalty_factor

        true_diff = higher_true - lower_true
        mask = tf.cast(tf.greater(true_diff, 0), tf.float32)

        loss = weighted_loss * mask
        num_valid_pairs = tf.reduce_sum(mask)

        return tf.cond(
            tf.greater(num_valid_pairs, 0),
            lambda: tf.reduce_sum(loss) / num_valid_pairs,
            lambda: tf.constant(0.0, dtype=tf.float32)
        )


class PeptideRankingConfig:
    def __init__(self):
        self.peptide_data_csv = "ACPs.csv"
        self.pretrained_config_path = "models/config.json"
        self.pretrained_weights_path = "models/peptide_mlm_model_best.weights.h5"
        self.model_path = "models/peptide_ranking_model"

        self.max_length = 33
        self.batch_size = 64
        self.test_split = 0.1
        self.validation_split = 0.1

        self.vocab_size = 25
        self.hidden_size = 512
        self.num_hidden_layers = 6
        self.num_attention_heads = 8
        self.intermediate_size = 2048
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.embedding_dim = 512

        self.learning_rate = 1e-5
        self.epochs = 50
        self.margin = 0.1

        self.false_high_penalty = 5.0
        self.false_low_penalty = 2.0
        self.activity_threshold = 4.3010

        self.rank_diff_factor = 1.5
        self.top_rank_focus = 2.0

        self.ext_train_sample_size = 50
        self.ext_test_top_k = 20

        self.aa_dict = None
        self.id_to_aa = None

    def load_from_pretrained(self):
        with open(self.pretrained_config_path, 'r') as f:
            pretrained_config = json.load(f)

        self.aa_dict = {}
        for k, v in pretrained_config['aa_dict'].items():
            self.aa_dict[k] = int(v)

        self.id_to_aa = {v: k for k, v in self.aa_dict.items()}

        self.max_length = pretrained_config['max_length']
        self.hidden_size = pretrained_config['hidden_size']
        self.num_hidden_layers = pretrained_config['num_hidden_layers']
        self.num_attention_heads = pretrained_config['num_attention_heads']
        self.intermediate_size = pretrained_config['intermediate_size']
        self.hidden_dropout_prob = pretrained_config['hidden_dropout_prob']
        self.attention_probs_dropout_prob = pretrained_config['attention_probs_dropout_prob']
        self.embedding_dim = pretrained_config['embedding_dim']
        self.vocab_size = pretrained_config['vocab_size']

        print(f"Loaded parameters from pretrained config: max_length={self.max_length}, hidden_size={self.hidden_size}")
        return self

    def save(self, filepath):
        with open(filepath, 'w') as f:
            config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k != 'id_to_aa'}
            json.dump(config_dict, f, indent=2)


class PeptideRankingDataProcessor:
    def __init__(self, config):
        self.config = config
        self.peptide_df = None
        self.peptide_sequences = []
        self.activity_scores = []

        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.peptide_names = []

    def load_data(self):
        print("Loading peptide data from CSV...")

        self.peptide_df = pd.read_csv(self.config.peptide_data_csv)

        sequences = []
        scores = []
        names = []

        for _, row in self.peptide_df.iterrows():
            seq = row['SEQUENCE']
            score = row['Activity_Score']
            name = row['NAME']

            if len(seq) <= self.config.max_length - 2:
                sequences.append(seq)
                scores.append(score)
                names.append(name)

        self.peptide_sequences = sequences
        self.activity_scores = np.array(scores)
        self.peptide_names = names

        print(f"Loaded {len(sequences)} peptide sequences with activity scores")

        return sequences, scores

    def split_data(self):
        n_samples = len(self.peptide_sequences)
        indices = list(range(n_samples))

        random.shuffle(indices)

        test_size = int(n_samples * self.config.test_split)
        val_size = int(n_samples * self.config.validation_split)

        self.test_indices = indices[:test_size]
        self.val_indices = indices[test_size:test_size + val_size]
        self.train_indices = indices[test_size + val_size:]

        print(f"Split data: {len(self.train_indices)} train, {len(self.val_indices)} validation, {len(self.test_indices)} test")

        self.prepare_data()

    def encode_sequence(self, sequence):
        tokens = [self.config.aa_dict['[CLS]']]

        for aa in sequence:
            if aa in self.config.aa_dict:
                tokens.append(self.config.aa_dict[aa])
            else:
                tokens.append(self.config.aa_dict['[UNK]'])

        tokens.append(self.config.aa_dict['[SEP]'])

        padding_length = self.config.max_length - len(tokens)
        if padding_length > 0:
            tokens.extend([self.config.aa_dict['[PAD]']] * padding_length)

        return tokens

    def prepare_data(self):
        all_encoded = np.array([self.encode_sequence(seq) for seq in self.peptide_sequences])

        self.X_train = all_encoded[self.train_indices]
        self.y_train = self.activity_scores[self.train_indices].reshape(-1, 1)

        self.X_val = all_encoded[self.val_indices]
        self.y_val = self.activity_scores[self.val_indices].reshape(-1, 1)

        self.X_test = all_encoded[self.test_indices]
        self.y_test = self.activity_scores[self.test_indices].reshape(-1, 1)

        print(f"Prepared encoded data - Train: {len(self.X_train)}, Validation: {len(self.X_val)}, Test: {len(self.X_test)}")

    def generate_cost_aware_pairs(self, X, y, threshold=4.3010):
        n_samples = len(X)
        pairs_X = []
        pairs_y = []

        active_indices = [i for i, score in enumerate(y.flatten()) if score > threshold]
        inactive_indices = [i for i, score in enumerate(y.flatten()) if score <= threshold]

        active_sorted = sorted([(i, y[i][0]) for i in active_indices], key=lambda x: x[1], reverse=True)

        top_active = [i for i, _ in active_sorted[:len(active_sorted)//4]] if active_sorted else []
        high_active = [i for i, _ in active_sorted[len(active_sorted)//4:len(active_sorted)//2]] if active_sorted else []
        other_active = [i for i, _ in active_sorted[len(active_sorted)//2:]] if active_sorted else []

        print(f"Sample distribution: top_active={len(top_active)}, high_active={len(high_active)}, other_active={len(other_active)}, inactive={len(inactive_indices)}")

        top_vs_inactive_count = int(n_samples * 1.2)
        for _ in range(top_vs_inactive_count):
            if top_active and inactive_indices:
                i = random.choice(top_active)
                j = random.choice(inactive_indices)
                pairs_X.append([X[i], X[j]])
                pairs_y.append([y[i], y[j]])

        top_vs_high_count = int(n_samples * 0.8)
        added_pairs = 0
        max_attempts = top_vs_high_count * 50
        attempts = 0

        while added_pairs < top_vs_high_count and attempts < max_attempts and top_active and high_active:
            i = random.choice(top_active)
            j = random.choice(high_active)
            attempts += 1

            if y[i][0] != y[j][0]:
                pairs_X.append([X[i], X[j]])
                pairs_y.append([y[i], y[j]])
                added_pairs += 1

        top_vs_other_count = int(n_samples * 0.5)
        added_pairs = 0
        max_attempts = top_vs_other_count * 50
        attempts = 0

        while added_pairs < top_vs_other_count and attempts < max_attempts and top_active and other_active:
            i = random.choice(top_active)
            j = random.choice(other_active)
            attempts += 1

            if y[i][0] != y[j][0]:
                pairs_X.append([X[i], X[j]])
                pairs_y.append([y[i], y[j]])
                added_pairs += 1

        high_vs_inactive_count = int(n_samples * 0.5)
        for _ in range(high_vs_inactive_count):
            if high_active and inactive_indices:
                i = random.choice(high_active)
                j = random.choice(inactive_indices)
                pairs_X.append([X[i], X[j]])
                pairs_y.append([y[i], y[j]])

        random_pairs_count = int(n_samples * 0.3)
        for _ in range(random_pairs_count):
            i, j = random.sample(range(n_samples), 2)
            if y[i][0] > y[j][0]:
                pairs_X.append([X[i], X[j]])
                pairs_y.append([y[i], y[j]])
            elif y[j][0] > y[i][0]:
                pairs_X.append([X[j], X[i]])
                pairs_y.append([y[j], y[i]])

        pairs_X = np.array(pairs_X) if pairs_X else np.empty((0, 2, X.shape[1]), dtype=X.dtype)
        pairs_y = np.array(pairs_y) if pairs_y else np.empty((0, 2, 1), dtype=y.dtype)

        if len(pairs_X) > 0:
            indices = list(range(len(pairs_X)))
            random.shuffle(indices)
            pairs_X = pairs_X[indices]
            pairs_y = pairs_y[indices]

        print(f"Generated {len(pairs_X)} sample pairs")
        return pairs_X, pairs_y

    def get_cost_aware_train_dataset(self):
        X_pairs, y_pairs = self.generate_cost_aware_pairs(
            self.X_train, self.y_train,
            threshold=self.config.activity_threshold
        )

        if len(X_pairs) == 0:
            print("Warning: No valid pairs generated, using backup method")
            return self.get_train_pairs_dataset()

        X_flat = X_pairs.reshape(-1, self.config.max_length)
        y_flat = y_pairs.reshape(-1, 1)

        dataset = tf.data.Dataset.from_tensor_slices((X_flat, y_flat))
        dataset = dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset, X_pairs, y_pairs

    def get_train_pairs_dataset(self):
        X_pairs, y_pairs = self.generate_pairs(self.X_train, self.y_train)

        X_flat = X_pairs.reshape(-1, self.config.max_length)
        y_flat = y_pairs.reshape(-1, 1)

        dataset = tf.data.Dataset.from_tensor_slices((X_flat, y_flat))
        dataset = dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset, X_pairs, y_pairs

    def generate_pairs(self, X, y):
        n_samples = len(X)
        pairs_X = []
        pairs_y = []

        for _ in range(n_samples * 2):
            i, j = random.sample(range(n_samples), 2)

            if y[i] > y[j]:
                pairs_X.append([X[i], X[j]])
                pairs_y.append([y[i], y[j]])
            elif y[j] > y[i]:
                pairs_X.append([X[j], X[i]])
                pairs_y.append([y[j], y[i]])

        pairs_X = np.array(pairs_X)
        pairs_y = np.array(pairs_y)

        indices = list(range(len(pairs_X)))
        random.shuffle(indices)
        pairs_X = pairs_X[indices]
        pairs_y = pairs_y[indices]

        return pairs_X, pairs_y

    def get_val_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        dataset = dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset, self.X_val, self.y_val

    def get_test_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        dataset = dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset, self.X_test, self.y_test


def calculate_top_k_overlap(y_true, y_pred, k=20, percentage=False):
    if k > len(y_true):
        k = len(y_true)

    true_top_k_idx = np.argsort(-y_true)[:k]
    pred_top_k_idx = np.argsort(-y_pred)[:k]

    overlap_count = len(set(true_top_k_idx) & set(pred_top_k_idx))

    if percentage:
        return overlap_count / k * 100
    else:
        return overlap_count, k


def evaluate_ranking_displacement(model, X, y, threshold=4.3010):
    y_pred = model.predict(X).flatten()
    y_true = y.flatten()

    true_ranks = np.argsort(np.argsort(-y_true))
    pred_ranks = np.argsort(np.argsort(-y_pred))

    active_mask = y_true > threshold

    front_pushed_back = 0
    n_front = int(len(y_true) * 0.2)

    for i in range(len(y_true)):
        if true_ranks[i] < n_front:
            rank_drop = max(0, pred_ranks[i] - true_ranks[i])
            severity = 1.0 + 2.0 * float(active_mask[i])
            front_pushed_back += rank_drop * severity / len(y_true)

    back_pushed_front = 0
    n_back = int(len(y_true) * 0.8)

    for i in range(len(y_true)):
        if true_ranks[i] >= n_back:
            rank_rise = max(0, true_ranks[i] - pred_ranks[i])
            severity = 1.0 + 4.0 * float(not active_mask[i])
            back_pushed_front += rank_rise * severity / len(y_true)

    max_possible_displacement = len(y_true) - 1
    front_pushed_back_error = front_pushed_back / max_possible_displacement
    back_pushed_front_error = back_pushed_front / max_possible_displacement

    total_displacement_error = (front_pushed_back_error * 0.6) + (back_pushed_front_error * 0.4)

    return {
        "rank_displacement_error": total_displacement_error,
        "front_pushed_back_error": front_pushed_back_error,
        "back_pushed_front_error": back_pushed_front_error
    }


def create_model(config):
    model = PeptideRankingModel(config)

    dummy_input = tf.ones((2, config.max_length), dtype=tf.int32)
    _ = model(dummy_input, training=False)

    print("Loading pretrained weights...")
    try:
        pretrained_model = PeptideMLMModel(config)

        _ = pretrained_model(dummy_input, training=False)

        pretrained_model.load_weights(config.pretrained_weights_path)
        print("Pretrained weights loaded successfully")

        model.embedding.set_weights(pretrained_model.embedding.get_weights())
        model.position_embedding.set_weights(pretrained_model.position_embedding.get_weights())

        for i in range(config.num_hidden_layers):
            model.encoder_layers[i].mha.set_weights(pretrained_model.encoder_layers[i].mha.get_weights())
            model.encoder_layers[i].ffn.set_weights(pretrained_model.encoder_layers[i].ffn.get_weights())
            model.encoder_layers[i].layernorm1.set_weights(pretrained_model.encoder_layers[i].layernorm1.get_weights())
            model.encoder_layers[i].layernorm2.set_weights(pretrained_model.encoder_layers[i].layernorm2.get_weights())

        model.layernorm.set_weights(pretrained_model.layernorm.get_weights())

        print("Pretrained weights successfully transferred to ranking model")

        del pretrained_model
        gc.collect()

    except Exception as e:
        print(f"Failed to load pretrained weights: {str(e)}")
        print("Will train with randomly initialized weights")

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=CostSensitiveRankingLoss(
            threshold=config.activity_threshold,
            false_high_penalty=config.false_high_penalty,
            false_low_penalty=config.false_low_penalty,
            rank_diff_factor=config.rank_diff_factor,
            top_rank_focus=config.top_rank_focus,
            margin=config.margin
        )
    )

    return model


def evaluate_ranking(model, X, y, names=None, top_k=10, ef_percentage=0.1, threshold=4.3010):
    y_pred = model.predict(X).flatten()
    y_true = y.flatten()

    nan_mask = np.isnan(y_pred)
    if np.any(nan_mask):
        print(f"Warning: Found {np.sum(nan_mask)} NaN predictions, replaced with 0")
        y_pred[nan_mask] = 0.0

    sorted_indices = np.argsort(-y_pred)

    k_5_percent = max(1, int(len(y_true) * 0.05))
    k_10_percent = max(1, int(len(y_true) * 0.10))
    k_20_percent = max(1, int(len(y_true) * 0.20))

    ndcg_5_percent = ndcg_score([y_true], [y_pred], k=k_5_percent)
    ndcg_10_percent = ndcg_score([y_true], [y_pred], k=k_10_percent)
    ndcg_20_percent = ndcg_score([y_true], [y_pred], k=k_20_percent)

    activity_threshold = np.percentile(y_true, 80)
    total_active = np.sum(y_true >= activity_threshold)

    cutoff = int(len(y_pred) * ef_percentage)
    top_indices = sorted_indices[:cutoff]

    active_in_selection = np.sum(y_true[top_indices] >= activity_threshold)

    random_expectation = total_active * ef_percentage
    ef = active_in_selection / random_expectation if random_expectation > 0 else 0

    binary_true = (y_true >= activity_threshold).astype(int)
    auc = roc_auc_score(binary_true, y_pred)

    spearman_corr, spearman_p_value = spearmanr(y_true, y_pred)

    threshold_relevance = np.percentile(y_true, 80)
    binary_relevance = (y_true >= threshold_relevance).astype(int)
    map_score = average_precision_score(binary_relevance, y_pred)

    top_indices = sorted_indices[:top_k] if len(sorted_indices) >= top_k else sorted_indices

    top_peptides = []
    for idx in top_indices:
        if names is not None:
            peptide_info = {
                "name": names[idx],
                "true_score": float(y_true[idx]),
                "predicted_score": float(y_pred[idx]),
                "rank": np.where(sorted_indices == idx)[0][0] + 1,
                "true_rank": np.where(np.argsort(-y_true) == idx)[0][0] + 1
            }
            top_peptides.append(peptide_info)

    actual_k_values = {
        "k_5_percent": k_5_percent,
        "k_10_percent": k_10_percent,
        "k_20_percent": k_20_percent
    }

    true_active = y_true > threshold
    pred_active = y_pred > 0.0

    boundary_accuracy = np.mean(true_active == pred_active)

    critical_errors = 0
    for i in range(len(y_true)):
        if not true_active[i]:
            for j in range(len(y_true)):
                if true_active[j] and y_pred[i] > y_pred[j]:
                    critical_errors += 1
                    break

    critical_error_rate = critical_errors / max(1, np.sum(~true_active))

    active_indices = np.where(true_active)[0]
    if len(active_indices) > 1:
        active_true = y_true[active_indices]
        active_pred = y_pred[active_indices]
        active_ndcg = ndcg_score([active_true], [active_pred], k=min(len(active_indices), top_k))
    else:
        active_ndcg = 1.0

    displacement_metrics = evaluate_ranking_displacement(
        model, X, y, threshold=threshold
    )

    top_20_overlap_count, actual_top_20 = calculate_top_k_overlap(y_true, y_pred, k=20, percentage=False)

    top_20_overlap_percentage = top_20_overlap_count / actual_top_20 * 100

    result = {
        "ndcg@5%": ndcg_5_percent,
        "ndcg@10%": ndcg_10_percent,
        "ndcg@20%": ndcg_20_percent,
        "actual_k_values": actual_k_values,
        "enrichment_factor": ef,
        "roc_auc": auc,
        "map_score": map_score,
        "spearman_correlation": spearman_corr,
        "top_peptides": top_peptides,
        "boundary_accuracy": boundary_accuracy,
        "critical_error_rate": critical_error_rate,
        "active_ndcg": active_ndcg,
        "top_20_overlap": {
            "count": top_20_overlap_count,
            "total": actual_top_20,
            "percentage": top_20_overlap_percentage
        },
    }

    result.update(displacement_metrics)

    return result


def train_ranking_model(config):
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    config.save(os.path.join(os.path.dirname(config.model_path), "ranking_config.json"))

    data_processor = PeptideRankingDataProcessor(config)
    data_processor.load_data()
    data_processor.split_data()

    model = create_model(config)

    val_dataset, X_val, y_val = data_processor.get_val_dataset()
    test_dataset, X_test, y_test = data_processor.get_test_dataset()

    checkpoint_path = f"{config.model_path}_best.weights.h5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        mode='min'
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(os.path.dirname(config.model_path), "training_log.csv")
    )

    metrics_csv_path = os.path.join(os.path.dirname(config.model_path), "training_metrics.csv")

    metrics_header = [
        "epoch", "loss", "ndcg@5%", "ndcg@10%", "ndcg@20%",
        "enrichment_factor", "roc_auc", "map_score", "spearman_correlation",
        "rank_displacement_error", "front_pushed_back_error", "back_pushed_front_error",
        "top_20_overlap_percentage"
    ]

    with open(metrics_csv_path, 'w') as f:
        f.write(','.join(metrics_header) + '\n')

    all_history = {
        'loss': [],
        'ranking_metrics': []
    }

    best_combined_score = 0
    best_epoch = 0

    print("Starting model training...")
    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs}")

        train_dataset, X_pairs, y_pairs = data_processor.get_cost_aware_train_dataset()
        print(f"Generated {len(X_pairs)} training pairs")

        history = model.fit(
            train_dataset,
            epochs=1,
            verbose=1,
            callbacks=[csv_logger]
        )

        for key in ['loss']:
            if key in history.history:
                if key in all_history:
                    all_history[key].extend(history.history[key])
                else:
                    all_history[key] = history.history[key]

        val_metrics = evaluate_ranking(model, X_val, y_val,
                                    names=[data_processor.peptide_names[i] for i in data_processor.val_indices],
                                    threshold=config.activity_threshold)

        all_history['ranking_metrics'].append(val_metrics)

        print(f"Epoch {epoch+1} validation metrics:")
        print(f"  Loss: {history.history['loss'][0]:.4f}")
        print(f"  NDCG@5%: {val_metrics['ndcg@5%']:.4f}")
        print(f"  NDCG@10%: {val_metrics['ndcg@10%']:.4f}")
        print(f"  NDCG@20%: {val_metrics['ndcg@20%']:.4f}")
        print(f"  Enrichment Factor: {val_metrics['enrichment_factor']:.4f}")
        print(f"  ROC AUC: {val_metrics['roc_auc']:.4f}")
        print(f"  MAP Score: {val_metrics['map_score']:.4f}")
        print(f"  Spearman Correlation: {val_metrics['spearman_correlation']:.4f}")
        print(f"  Rank Displacement Error: {val_metrics['rank_displacement_error']:.4f}")
        print(f"    - Front Pushed Back Error: {val_metrics['front_pushed_back_error']:.4f}")
        print(f"    - Back Pushed Front Error: {val_metrics['back_pushed_front_error']:.4f}")
        print(f"  Top-20 Overlap: {val_metrics['top_20_overlap']['percentage']:.2f}%")

        with open(metrics_csv_path, 'a') as f:
            metrics_row = [
                epoch + 1,
                history.history['loss'][0],
                val_metrics['ndcg@5%'],
                val_metrics['ndcg@10%'],
                val_metrics['ndcg@20%'],
                val_metrics['enrichment_factor'],
                val_metrics['roc_auc'],
                val_metrics['map_score'],
                val_metrics['spearman_correlation'],
                val_metrics['rank_displacement_error'],
                val_metrics['front_pushed_back_error'],
                val_metrics['back_pushed_front_error'],
                val_metrics['top_20_overlap']['percentage']
            ]
            f.write(','.join(map(str, metrics_row)) + '\n')

        current_combined_score = val_metrics['map_score'] * 0.5 + val_metrics['spearman_correlation'] * 0.5

        if current_combined_score > best_combined_score:
            best_combined_score = current_combined_score
            best_epoch = epoch + 1

            model.save_weights(checkpoint_path)
            print(f"Saved new best model: Combined Score improved to {best_combined_score:.4f}")
            print(f"  MAP Score: {val_metrics['map_score']:.4f}, Spearman: {val_metrics['spearman_correlation']:.4f}")

    model.save_weights(f"{config.model_path}_final.weights.h5")

    if os.path.exists(checkpoint_path):
        print(f"Loading best model weights (from epoch {best_epoch})")
        model.load_weights(checkpoint_path)

    test_metrics = evaluate_ranking(
        model, X_test, y_test,
        names=[data_processor.peptide_names[i] for i in data_processor.test_indices],
        threshold=config.activity_threshold
    )
    y_pred_test = model.predict(X_test).flatten()

    print("\n" + "="*50)
    print("Final test set performance:")
    print(f"NDCG@5% (k={test_metrics['actual_k_values']['k_5_percent']}): {test_metrics['ndcg@5%']:.4f}")
    print(f"NDCG@10% (k={test_metrics['actual_k_values']['k_10_percent']}): {test_metrics['ndcg@10%']:.4f}")
    print(f"NDCG@20% (k={test_metrics['actual_k_values']['k_20_percent']}): {test_metrics['ndcg@20%']:.4f}")
    print(f"Enrichment Factor: {test_metrics['enrichment_factor']:.4f}")
    print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
    print(f"MAP Score: {test_metrics['map_score']:.4f}")
    print(f"Spearman Correlation: {test_metrics['spearman_correlation']:.4f}")
    print(f"Rank Displacement Error: {test_metrics['rank_displacement_error']:.4f}")
    print(f"  - Front Pushed Back Error: {test_metrics['front_pushed_back_error']:.4f}")
    print(f"  - Back Pushed Front Error: {test_metrics['back_pushed_front_error']:.4f}")
    print(f"Top-20 Overlap: {test_metrics['top_20_overlap']['count']}/{test_metrics['top_20_overlap']['total']} " +
        f"({test_metrics['top_20_overlap']['percentage']:.2f}%)")
    print("="*50)

    print("\nTop 5 peptides by predicted activity:")
    for i, peptide in enumerate(test_metrics['top_peptides'][:5]):
        print(f"{i+1}. {peptide['name']} - True: {peptide['true_score']:.4f}, Pred: {peptide['predicted_score']:.4f}")
        print(f"   True Rank: {peptide['true_rank']}, Predicted Rank: {peptide['rank']}")

    with open(os.path.join(os.path.dirname(config.model_path), "training_history.json"), "w") as f:
        def convert_numpy(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        history_serializable = convert_numpy(all_history)
        json.dump(history_serializable, f, indent=2)

    results = {
        "best_epoch": best_epoch,
        "test_metrics": test_metrics
    }

    with open(os.path.join(os.path.dirname(config.model_path), "test_results.json"), "w") as f:
        def convert_numpy(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        serializable_results = convert_numpy(results)
        json.dump(serializable_results, f, indent=2)

    with open(os.path.join(os.path.dirname(config.model_path), "test_results_with_external.json"), "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    return model, all_history, results


def main():
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    start_time = time.time()
    print("="*50)
    print("Starting Peptide Activity Ranking Model Training with Enhanced Cost-Aware Optimization")
    print("="*50)

    config = PeptideRankingConfig()
    config.load_from_pretrained()

    model, history, results = train_ranking_model(config)

    total_time = time.time() - start_time
    print(f"Done! Total time: {total_time//3600} hours {(total_time%3600)//60} minutes {total_time%60:.2f} seconds")
    print(f"Best model from Epoch {results['best_epoch']}")
    print(f"NDCG@10%: {results['test_metrics']['ndcg@10%']:.4f}")
    print(f"MAP Score: {results['test_metrics']['map_score']:.4f}")
    print(f"Spearman Correlation: {results['test_metrics']['spearman_correlation']:.4f}")
    print(f"Rank Displacement Error: {results['test_metrics']['rank_displacement_error']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()

