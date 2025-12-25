#!/usr/bin/env python3

import os
import sys
import mysql.connector
import requests
from collections import defaultdict, Counter
import argparse
import numpy as np
import math
import csv
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO
import shutil
import json
import pickle

class ProteaseAnalyzer:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None
        self.cursor = None
        self.base_output_dir = "protease_analysis_batch"
        self.current_output_dir = None
        self.model_cache_file = os.path.join(self.base_output_dir, "prediction_models.pkl")
        self.sequence_cache_dir = os.path.join(self.base_output_dir, "sequence_cache")
        os.makedirs(self.sequence_cache_dir, exist_ok=True)
        self.local_sequences = {}
        self.local_fasta_loaded = False
        self.frequency_matrix = None
        self.position_weights = None
        self.min_tp_score = None
        self.window_size = 10
        os.makedirs(self.base_output_dir, exist_ok=True)

    def connect_to_db(self):
        try:
            self.connection = mysql.connector.connect(**self.db_config, use_pure=True)
            self.cursor = self.connection.cursor(dictionary=True)
            print(f"Successfully connected to database {self.db_config['database']}")
            return True
        except Exception as err:
            print(f"Database connection error: {type(err).__name__}: {str(err)}")
            return False

    def close_connection(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def set_protease_output_dir(self, uniprot_id):
        self.current_output_dir = os.path.join(self.base_output_dir, uniprot_id)
        os.makedirs(self.current_output_dir, exist_ok=True)
        return self.current_output_dir

    def get_protease_substrates(self, code):
        query = """
        SELECT code, uniprot_acc, p1, cleavage_type, cleavage_evidence
        FROM cleavage
        WHERE code = %s AND uniprot_acc IS NOT NULL
        ORDER BY uniprot_acc, p1
        """

        try:
            self.cursor.execute(query, (code,))
            results = self.cursor.fetchall()

            if not results:
                print(f"No substrate data found for protease {code}")
                return {}

            substrates = defaultdict(list)
            for row in results:
                substrates[row['uniprot_acc']].append(row)

            print(f"Found {len(substrates)} substrates with {len(results)} cleavage sites")
            return substrates

        except Exception as err:
            print(f"Query error: {type(err).__name__}: {str(err)}")
            return {}

    def load_local_proteome(self, fasta_file="uniprotkb_organism_id_9606_2025_05_13.fasta"):
        if self.local_fasta_loaded:
            return

        if not os.path.exists(fasta_file):
            print(f"Warning: Local proteome file {fasta_file} not found")
            self.local_fasta_loaded = True
            return

        try:
            print(f"Loading sequences from local proteome file: {fasta_file}")
            count = 0

            for record in SeqIO.parse(fasta_file, "fasta"):
                if '|' in record.id:
                    uniprot_id = record.id.split('|')[1]
                elif record.id.startswith("UP") or record.id.startswith("sp") or record.id.startswith("tr"):
                    parts = record.id.split()
                    uniprot_id = parts[0].strip()
                else:
                    uniprot_id = record.id.strip()

                self.local_sequences[uniprot_id] = (record.description, str(record.seq))

                cache_file = os.path.join(self.sequence_cache_dir, f"{uniprot_id}.fasta")
                if not os.path.exists(cache_file):
                    with open(cache_file, 'w') as f:
                        f.write(f">{record.description}\n")
                        f.write(str(record.seq))

                count += 1

            print(f"Loaded {count} sequences from local proteome file")
            self.local_fasta_loaded = True

        except Exception as e:
            print(f"Error loading local proteome file: {e}")
            self.local_fasta_loaded = True

    def get_uniprot_sequence(self, uniprot_id):
        if not self.local_fasta_loaded:
            self.load_local_proteome()

        cache_file = os.path.join(self.sequence_cache_dir, f"{uniprot_id}.fasta")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    content = f.read()

                header = ""
                sequence = ""
                for i, line in enumerate(content.split('\n')):
                    if i == 0 and line.startswith('>'):
                        header = line
                    elif not line.startswith('>'):
                        sequence += line.strip()

                if not header.startswith('>'):
                    header = '>' + header

                return header, sequence
            except Exception as e:
                print(f"Error reading cached sequence for {uniprot_id}: {e}")

        if uniprot_id in self.local_sequences:
            header, sequence = self.local_sequences[uniprot_id]
            print(f"Found {uniprot_id} in local proteome")

            header_with_prefix = '>' + header if not header.startswith('>') else header
            return header_with_prefix, sequence

        print(f"Sequence for {uniprot_id} not found locally, trying UniProt...")
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                sequence = ""
                header = ""
                for i, line in enumerate(response.text.split('\n')):
                    if i == 0 and line.startswith('>'):
                        header = line
                    elif not line.startswith('>'):
                        sequence += line.strip()

                with open(cache_file, 'w') as f:
                    f.write(response.text)

                header_stripped = header[1:] if header.startswith('>') else header
                self.local_sequences[uniprot_id] = (header_stripped, sequence)

                return header, sequence
            else:
                print(f"Unable to retrieve sequence for {uniprot_id}, status code: {response.status_code}")
                return None, None

        except Exception as e:
            print(f"Error downloading sequence for {uniprot_id}: {e}")
            return None, None

    def extract_position_specific_aas(self, exp_sites, window=5):
        positions = {}
        for i in range(-window, window+1):
            if i < 0:
                pos_name = f"P{abs(i)}"
            elif i > 0:
                pos_name = f"P{i}'"
            else:
                continue
            positions[pos_name] = []

        full_peptides = []

        for site in exp_sites:
            context = site['context']
            arrow_pos = context.find('↓')
            if arrow_pos == -1:
                continue

            peptide_context = context.replace('↓', '')
            if len(peptide_context) >= 10:
                full_peptides.append(peptide_context)

            for i in range(-window, window+1):
                if i < 0:
                    pos_name = f"P{abs(i)}"
                    aa_pos = arrow_pos + i
                elif i > 0:
                    pos_name = f"P{i}'"
                    aa_pos = arrow_pos + i - 1
                else:
                    continue

                if 0 <= aa_pos < len(context.replace('↓', '')):
                    aa = context.replace('↓', '')[aa_pos]
                    positions[pos_name].append(aa)
                else:
                    positions[pos_name].append('X')

        return positions, full_peptides

    def analyze_position_properties(self, positions):
        position_properties = {}

        for pos, aas in positions.items():
            valid_aas = [aa for aa in aas if aa in "ACDEFGHIKLMNPQRSTVWY"]

            if not valid_aas:
                continue

            aa_frequency = Counter(valid_aas)

            position_properties[pos] = {
                'valid_aas': valid_aas,
                'aa_count': len(valid_aas),
                'aa_frequency': aa_frequency
            }

        return position_properties

    def calculate_peptide_hydrophobicity(self, peptides):
        hydrophobicities = []

        for peptide in peptides:
            valid_peptide = ''.join([aa for aa in peptide if aa in "ACDEFGHIKLMNPQRSTVWY"])
            if len(valid_peptide) >= 5:
                try:
                    analyzer = ProteinAnalysis(valid_peptide)
                    hydrophobicity = analyzer.gravy()
                    hydrophobicities.append(hydrophobicity)
                except Exception as e:
                    print(f"Error calculating hydrophobicity for peptide {valid_peptide}: {e}")

        if hydrophobicities:
            avg_hydrophobicity = np.mean(hydrophobicities)
            std_hydrophobicity = np.std(hydrophobicities)
            return avg_hydrophobicity, std_hydrophobicity, len(hydrophobicities)
        else:
            return None, None, 0

    def build_frequency_matrix(self, position_properties):
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        positions = ["P5", "P4", "P3", "P2", "P1", "P1'", "P2'", "P3'", "P4'", "P5'"]

        frequency_matrix = {}
        for aa in amino_acids:
            frequency_matrix[aa] = {}
            for pos in positions:
                frequency_matrix[aa][pos] = 0.0

        for pos in positions:
            if pos in position_properties:
                props = position_properties[pos]
                total_count = props['aa_count']

                if total_count > 0:
                    for aa in amino_acids:
                        count = props['aa_frequency'].get(aa, 0)
                        frequency_matrix[aa][pos] = count / total_count

        return frequency_matrix, positions

    def calculate_position_entropy(self, frequency_matrix, positions):
        entropy_values = {}

        for pos in positions:
            entropy = 0
            for aa in "ACDEFGHIKLMNPQRSTVWY":
                freq = frequency_matrix[aa][pos]
                if freq > 0:
                    entropy -= freq * math.log2(freq)
            entropy_values[pos] = entropy

        max_entropy = max(entropy_values.values()) if entropy_values else 0
        position_weights = {}

        for pos in positions:
            if max_entropy > 0:
                position_weights[pos] = (max_entropy - entropy_values[pos]) / max_entropy
            else:
                position_weights[pos] = 0

        weight_sum = sum(position_weights.values())
        if weight_sum > 0:
            normalized_weights = {pos: w / weight_sum for pos, w in position_weights.items()}
        else:
            normalized_weights = {pos: 0.1 for pos in positions}

        return entropy_values, normalized_weights

    def score_cleavage_site(self, sequence, position, frequency_matrix, position_weights):
        positions = ["P5", "P4", "P3", "P2", "P1", "P1'", "P2'", "P3'", "P4'", "P5'"]

        start = max(0, position - 5)
        end = min(len(sequence), position + 5)

        window_seq = sequence[start:end]

        start_offset = max(0, 5 - position)

        score = 0
        used_weights_sum = 0

        for i, aa in enumerate(window_seq):
            if i + start_offset < 5:
                pos_idx = i + start_offset
                pos_name = f"P{5-pos_idx}"
            elif i + start_offset == 5:
                continue
            else:
                pos_idx = i + start_offset - 6
                pos_name = f"P{pos_idx+1}'"

            if pos_name in positions:
                aa_freq = frequency_matrix.get(aa, {}).get(pos_name, 0)
                position_weight = position_weights[pos_name]

                score += aa_freq * position_weight
                used_weights_sum += position_weight

        if used_weights_sum > 0:
            score = score / used_weights_sum

        return score

    def predict_cleavage_sites(self, protein_sequence, frequency_matrix, position_weights, threshold=None):
        predictions = []

        for pos in range(1, len(protein_sequence)):
            score = self.score_cleavage_site(
                protein_sequence, pos,
                frequency_matrix, position_weights
            )

            p1_position = pos
            predictions.append({
                'position': p1_position,
                'score': score,
                'pre_aa': protein_sequence[pos-1] if pos > 0 else None,
                'post_aa': protein_sequence[pos] if pos < len(protein_sequence) else None
            })

        if threshold is not None:
            predictions = [p for p in predictions if p['score'] >= threshold]

        predictions.sort(key=lambda x: x['score'], reverse=True)
        return predictions

    def evaluate_predictions(self, predictions, true_sites, protein_length, threshold=None):
        if threshold is not None:
            filtered_predictions = [p for p in predictions if p['score'] >= threshold]
        else:
            filtered_predictions = predictions

        predicted_positions = [p['position'] for p in filtered_predictions]

        true_positives = set(predicted_positions).intersection(true_sites)
        false_positives = set(predicted_positions) - set(true_sites)
        false_negatives = set(true_sites) - set(predicted_positions)

        precision = len(true_positives) / len(predicted_positions) if predicted_positions else 0
        recall = len(true_positives) / len(true_sites) if true_sites else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'protein_length': protein_length
        }

    def analyze_cleavage_sites(self, code):
        if not self.connect_to_db():
            return None, 0, 0

        substrates = self.get_protease_substrates(code)
        if not substrates:
            self.close_connection()
            return None, 0, 0

        cleavage_sites = {
            'experimental': [],
            'other': []
        }

        protein_sequences = {}

        site_distances = []
        protein_site_distances = defaultdict(list)

        report_file = os.path.join(self.current_output_dir, f"{code}_substrate_analysis.txt")
        with open(report_file, 'w') as f:
            f.write(f"# Comprehensive Cleavage Site Analysis for Protease {code}\n\n")
            f.write(f"Total substrates analyzed: {len(substrates)}\n\n")

            for uniprot_id, cleavages in substrates.items():
                print(f"Analyzing substrate: {uniprot_id}...")

                header, sequence = self.get_uniprot_sequence(uniprot_id)
                if not sequence:
                    f.write(f"## Substrate: {uniprot_id}\n")
                    f.write("Unable to retrieve sequence information\n\n")
                    continue

                protein_sequences[uniprot_id] = sequence

                f.write(f"## Substrate: {uniprot_id}\n\n")
                if header:
                    f.write(f"{header}\n")

                valid_sites = []
                valid_p1_positions = []

                for site in sorted(cleavages, key=lambda x: x['p1']):
                    p1 = site['p1']
                    site_index = p1 - 1

                    if site_index < 0 or site_index >= len(sequence):
                        continue

                    valid_sites.append((site_index, p1, site))
                    valid_p1_positions.append(p1)

                    start = max(0, site_index - 5)
                    end = min(len(sequence), site_index + 6)
                    context = sequence[start:site_index] + sequence[site_index] + "↓" + sequence[site_index+1:end]

                    site_info = {
                        'uniprot_id': uniprot_id,
                        'p1': p1,
                        'cleavage_type': site['cleavage_type'] or 'N/A',
                        'cleavage_evidence': site['cleavage_evidence'] or 'N/A',
                        'context': context
                    }

                    evidence_type = site['cleavage_evidence'] or 'unknown'

                    if evidence_type == 'experimental':
                        cleavage_sites['experimental'].append(site_info)
                    else:
                        cleavage_sites['other'].append(site_info)

                if len(valid_p1_positions) >= 2:
                    valid_p1_positions.sort()
                    for i in range(1, len(valid_p1_positions)):
                        distance = valid_p1_positions[i] - valid_p1_positions[i-1]
                        site_distances.append(distance)
                        protein_site_distances[uniprot_id].append({
                            'site1': valid_p1_positions[i-1],
                            'site2': valid_p1_positions[i],
                            'distance': distance
                        })

                if valid_sites:
                    f.write("### Complete Sequence (All Cleavage Sites):\n\n")

                    marked_sequence = list(sequence)
                    for site_index, p1, site_info in sorted(valid_sites, reverse=True):
                        marked_sequence.insert(site_index + 1, "↓")

                    marked_sequence = ''.join(marked_sequence)

                    for i in range(0, len(marked_sequence), 60):
                        f.write(marked_sequence[i:i+60] + "\n")

                    f.write("\n")

            exp_distances = {}
            exp_sites_per_protein = defaultdict(list)

            for site in cleavage_sites['experimental']:
                exp_sites_per_protein[site['uniprot_id']].append(site['p1'])

            for uniprot_id, sites in exp_sites_per_protein.items():
                if len(sites) >= 2:
                    sites.sort()
                    for i in range(len(sites)-1):
                        site1 = sites[i]
                        site2 = sites[i+1]
                        exp_distances[(uniprot_id, site1)] = site2 - site1

            exp_count = len(cleavage_sites['experimental'])
            unique_substrate_count = len(exp_sites_per_protein)

            if exp_count == 0:
                f.write("# No Experimentally Verified Cleavage Sites Found\n\n")
                f.write("This protease does not have any experimentally verified cleavage sites in the database.\n")
                f.write("Unable to build a prediction model without experimental data.\n")
                print(f"No experimental data for {code}, skipping model building")
                self.close_connection()
                return None, 0, 0

            f.write(f"# Experimentally Verified Cleavage Sites (Total: {exp_count})\n\n")

            position_aas, full_peptides = self.extract_position_specific_aas(cleavage_sites['experimental'])

            avg_hydro, std_hydro, peptide_count = self.calculate_peptide_hydrophobicity(full_peptides)

            position_properties = self.analyze_position_properties(position_aas)

            f.write("## Amino Acid Characteristics Around Cleavage Sites\n\n")

            f.write("### Amino Acid Frequency at Each Position\n\n")

            f.write("| Amino Acid |")
            for pos in ["P5", "P4", "P3", "P2", "P1", "P1'", "P2'", "P3'", "P4'", "P5'"]:
                f.write(f" {pos:^7} |")
            f.write("\n")

            f.write("|------------|")
            for _ in range(10):
                f.write("-------:|")
            f.write("\n")

            aa_list = "ACDEFGHIKLMNPQRSTVWY"
            for aa in aa_list:
                f.write(f"| **{aa}** |")
                for pos in ["P5", "P4", "P3", "P2", "P1", "P1'", "P2'", "P3'", "P4'", "P5'"]:
                    if pos in position_properties:
                        freq = position_properties[pos]['aa_frequency'].get(aa, 0)
                        total = position_properties[pos]['aa_count']
                        percent = (freq / total) * 100 if total > 0 else 0
                        f.write(f" {percent:6.2f}% |")
                    else:
                        f.write("       - |")
                f.write("\n")

            frequency_matrix, positions = self.build_frequency_matrix(position_properties)
            entropy_values, position_weights = self.calculate_position_entropy(frequency_matrix, positions)

            self.frequency_matrix = frequency_matrix
            self.position_weights = position_weights

            f.write("\n### Position Entropy and Weights\n\n")
            f.write("| Position | Entropy | Normalized Weight |\n")
            f.write("|----------|--------:|-----------------:|\n")

            for pos in positions:
                f.write(f"| {pos} | {entropy_values[pos]:.4f} | {position_weights[pos]:.4f} |\n")

            f.write("\n")

            all_predictions = []
            true_positive_scores = []
            all_scores = []

            for uniprot_id, sequence in protein_sequences.items():
                if uniprot_id not in exp_sites_per_protein:
                    continue

                true_sites = exp_sites_per_protein[uniprot_id]

                predictions = self.predict_cleavage_sites(
                    sequence, frequency_matrix, position_weights
                )

                all_scores.extend([p['score'] for p in predictions])

                for pred in predictions:
                    pred['uniprot_id'] = uniprot_id
                    pred['is_true_site'] = pred['position'] in true_sites
                    if pred['is_true_site']:
                        true_positive_scores.append(pred['score'])

                all_predictions.extend(predictions)

            if true_positive_scores:
                min_tp_score = min(true_positive_scores)
                self.min_tp_score = min_tp_score
                f.write(f"### Determined Threshold Score\n\n")
                f.write(f"The minimum score among true positive predictions is: **{min_tp_score:.6f}**\n")
                f.write(f"This value will be used as the threshold for predictions on unknown sequences.\n\n")
            else:
                f.write(f"### No True Positive Predictions\n\n")
                f.write(f"Could not identify any true positive predictions for the experimental sites.\n")
                f.write(f"The prediction model may not be reliable. Skipping this protease.\n\n")
                self.close_connection()
                return None, exp_count, unique_substrate_count

            f.write("## Detailed Cleavage Site Data\n\n")
            f.write("| UniProt ID | Position(p1) | Cleavage Type | Evidence | Sequence Context (±5aa) | Distance to Next | Protein Length | Prediction Score |\n")
            f.write("|------------|--------------|---------------|----------|-------------------------|-----------------|----------------|-----------------:|\n")

            sorted_sites = sorted(cleavage_sites['experimental'], key=lambda x: (x['uniprot_id'], x['p1']))

            for site in sorted_sites:
                uniprot_id = site['uniprot_id']
                p1 = site['p1']

                next_distance = exp_distances.get((uniprot_id, p1), "N/A")

                protein_length = len(protein_sequences.get(uniprot_id, ""))

                prediction_score = "N/A"
                for pred in all_predictions:
                    if pred['uniprot_id'] == uniprot_id and pred['position'] == p1:
                        prediction_score = f"{pred['score']:.4f}"
                        break

                f.write(f"| {uniprot_id} | {p1} | {site['cleavage_type']} | {site['cleavage_evidence']} | " +
                        f"{site['context']} | {next_distance} | {protein_length} | {prediction_score} |\n")

            f.write("\n")

            other_count = len(cleavage_sites['other'])
            if other_count > 0:
                f.write(f"# Other Evidence Cleavage Sites (Total: {other_count})\n\n")
                f.write("| UniProt ID | Position(p1) | Cleavage Type | Evidence | Sequence Context (±5aa) | Protein Length |\n")
                f.write("|------------|--------------|---------------|----------|-------------------------|----------------|\n")

                for site in sorted(cleavage_sites['other'], key=lambda x: (x['uniprot_id'], x['p1'])):
                    uniprot_id = site['uniprot_id']
                    protein_length = len(protein_sequences.get(uniprot_id, ""))

                    f.write(f"| {uniprot_id} | {site['p1']} | {site['cleavage_type']} | {site['cleavage_evidence']} | {site['context']} | {protein_length} |\n")

                f.write("\n")

            f.write("# Prediction Score Statistics\n\n")

            if true_positive_scores:
                f.write("## True Positive Score Statistics\n\n")

                tp_min = min(true_positive_scores)
                tp_max = max(true_positive_scores)
                tp_quartiles = np.percentile(true_positive_scores, [0, 25, 50, 75, 100])

                f.write(f"- Count: {len(true_positive_scores)}\n")
                f.write(f"- Minimum: {tp_min:.6f}\n")
                f.write(f"- Maximum: {tp_max:.6f}\n")
                f.write(f"- Quartile Distribution [0%, 25%, 50%, 75%, 100%]: [{tp_quartiles[0]:.6f}, {tp_quartiles[1]:.6f}, {tp_quartiles[2]:.6f}, {tp_quartiles[3]:.6f}, {tp_quartiles[4]:.6f}]\n\n")

            if all_scores:
                f.write("## All Prediction Sites Score Statistics\n\n")

                all_min = min(all_scores)
                all_max = max(all_scores)
                all_quartiles = np.percentile(all_scores, [0, 25, 50, 75, 100])

                f.write(f"- Count: {len(all_scores)}\n")
                f.write(f"- Minimum: {all_min:.6f}\n")
                f.write(f"- Maximum: {all_max:.6f}\n")
                f.write(f"- Quartile Distribution [0%, 25%, 50%, 75%, 100%]: [{all_quartiles[0]:.6f}, {all_quartiles[1]:.6f}, {all_quartiles[2]:.6f}, {all_quartiles[3]:.6f}, {all_quartiles[4]:.6f}]\n")

        prediction_model = {
            'code': code,
            'frequency_matrix': frequency_matrix if 'frequency_matrix' in locals() else None,
            'position_weights': position_weights if 'position_weights' in locals() else None,
            'min_tp_score': self.min_tp_score
        }

        print(f"Analysis for {code} complete. Report saved to {report_file}")
        return prediction_model, exp_count, unique_substrate_count

    def extract_peptides_from_fasta(self, fasta_file, remove_n_term=0):
        sequences = {}

        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                uniprot_id = record.id.split('|')[1] if '|' in record.id else record.id
                sequence = str(record.seq)

                if remove_n_term > 0 and len(sequence) > remove_n_term:
                    sequence = sequence[remove_n_term:]

                sequences[uniprot_id] = {
                    'header': record.description,
                    'sequence': sequence
                }

            print(f"Extracted {len(sequences)} sequences from {fasta_file}")
            return sequences
        except Exception as e:
            print(f"Error extracting sequences from {fasta_file}: {e}")
            return {}

    def predict_sites_for_protein(self, uniprot_id, sequence, model, output_file=None):
        if not model or not model['frequency_matrix'] or not model['position_weights']:
            print(f"Error: No valid prediction model for {uniprot_id}")
            return []

        threshold = model['min_tp_score'] if model['min_tp_score'] is not None else 0.05

        predictions = self.predict_cleavage_sites(
            sequence, model['frequency_matrix'], model['position_weights'], threshold
        )

        if output_file and predictions:
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)

            with open(output_file, 'w') as f:
                f.write(f"# Predicted Cleavage Sites for {uniprot_id} using model {model['code']}\n\n")
                f.write("| Position | Score | P1 AA | P1' AA | P5-P5' Context |\n")
                f.write("|----------|------:|-------|--------|---------------|\n")

                for pred in predictions:
                    position = pred['position']
                    p1_aa = pred['pre_aa']
                    p1_prime_aa = pred['post_aa']

                    start = max(0, position - 6)
                    end = min(len(sequence), position + 5)
                    context = sequence[start:position] + "↓" + sequence[position:end]

                    f.write(f"| {position} | {pred['score']:.6f} | {p1_aa} | {p1_prime_aa} | {context} |\n")

        return predictions

    def process_fasta_file(self, fasta_file, all_models, remove_n_term=0, peptide_output_file=None):
        sequences = self.extract_peptides_from_fasta(fasta_file, remove_n_term)

        if not sequences:
            print(f"No sequences found in {fasta_file}")
            return

        all_peptides = []

        for uniprot_id, seq_info in sequences.items():
            sequence = seq_info['sequence']

            for model_id, model in all_models.items():
                if not model or not model['frequency_matrix']:
                    continue

                print(f"Predicting cleavage sites for {uniprot_id} with protease {model_id}")

                merops_dir = os.path.join(self.base_output_dir, model_id)
                os.makedirs(merops_dir, exist_ok=True)

                output_file = os.path.join(merops_dir, f"{uniprot_id}_cleavage_sites.txt")

                predictions = self.predict_sites_for_protein(uniprot_id, sequence, model, output_file)

                for i, pred in enumerate(predictions):
                    if i < len(predictions) - 1:
                        start_pos = pred['position']
                        end_pos = predictions[i+1]['position']

                        if start_pos < end_pos:
                            peptide = sequence[start_pos:end_pos]

                            peptide_info = {
                                'substrate_uniprot': uniprot_id,
                                'protease_merops_id': model_id,
                                'start_pos': start_pos + 1,
                                'end_pos': end_pos,
                                'peptide': peptide,
                                'length': len(peptide),
                                'start_score': pred['score'],
                                'end_score': predictions[i+1]['score']
                            }
                            all_peptides.append(peptide_info)

        if peptide_output_file and all_peptides:
            output_dir = os.path.dirname(peptide_output_file)
            os.makedirs(output_dir, exist_ok=True)

            with open(peptide_output_file, 'w', newline='') as f:
                fieldnames = ['substrate_uniprot', 'protease_merops_id', 'start_pos', 'end_pos',
                            'peptide', 'length', 'start_score', 'end_score']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for peptide in all_peptides:
                    writer.writerow(peptide)

            print(f"Saved {len(all_peptides)} predicted peptides to {peptide_output_file}")

        return all_peptides

    def save_models(self, all_models):
        with open(self.model_cache_file, 'wb') as f:
            pickle.dump(all_models, f)
        print(f"Saved {len(all_models)} prediction models to {self.model_cache_file}")

    def load_models(self):
        if os.path.exists(self.model_cache_file):
            try:
                with open(self.model_cache_file, 'rb') as f:
                    all_models = pickle.load(f)
                print(f"Loaded {len(all_models)} prediction models from {self.model_cache_file}")
                return all_models
            except Exception as e:
                print(f"Error loading models: {e}")
                return None
        else:
            print(f"Model cache file {self.model_cache_file} not found")
            return None

    def organize_results_by_protease(self, missing_peptides, processed_peptides, uniprot_to_merops):
        all_peptides = []
        if missing_peptides:
            all_peptides.extend(missing_peptides)
        if processed_peptides:
            all_peptides.extend(processed_peptides)

        if not all_peptides:
            print("No peptides to organize")
            return

        protease_peptides = defaultdict(list)
        for peptide in all_peptides:
            protease_id = peptide['protease_merops_id']
            protease_peptides[protease_id].append(peptide)

        for protease_id, peptides in protease_peptides.items():
            uniprot_ids = [uniprotid for uniprotid, meropsid in uniprot_to_merops.items() if meropsid == protease_id]

            for uniprot_id in uniprot_ids:
                output_dir = os.path.join(self.base_output_dir, uniprot_id)
                os.makedirs(output_dir, exist_ok=True)

                output_file = os.path.join(output_dir, f"{protease_id}_peptides.csv")
                with open(output_file, 'w', newline='') as f:
                    fieldnames = ['substrate_uniprot', 'protease_merops_id', 'start_pos', 'end_pos',
                                'peptide', 'length', 'start_score', 'end_score']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for peptide in peptides:
                        writer.writerow(peptide)

                print(f"Saved {len(peptides)} peptides for protease {protease_id} to {output_file}")

    def batch_process_proteases(self, csv_file, analyze_only=False, predict_only=False):
        if predict_only:
            all_models = self.load_models()
            if not all_models:
                print("No saved models found. Cannot run prediction-only mode.")
                return

            uniprot_to_merops = {}
            try:
                with open(csv_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        uniprot_to_merops[row.get('uniprot_id', 'Unknown')] = row.get('merops_id', 'Unknown')
            except Exception as e:
                print(f"Error reading CSV file {csv_file}: {e}")
                return

            print(f"Skipping analysis phase. Running prediction using {len(all_models)} cached models.")
            missing_peptides_file = os.path.join(self.base_output_dir, "missing_sequences_peptides.csv")
            missing_peptides = self.process_fasta_file("missing_sequences.fasta", all_models,
                                                  remove_n_term=20,
                                                  peptide_output_file=missing_peptides_file)

            processed_peptides_file = os.path.join(self.base_output_dir, "processed_by_SignalP6_peptides.csv")
            processed_peptides = self.process_fasta_file("processed_by_SignalP6.fasta", all_models,
                                                    remove_n_term=0,
                                                    peptide_output_file=processed_peptides_file)

            self.organize_results_by_protease(missing_peptides, processed_peptides, uniprot_to_merops)
            return

        proteases = []
        uniprot_to_merops = {}

        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    proteases.append({
                        'gene_name': row.get('gene_name', 'Unknown'),
                        'uniprot_id': row.get('uniprot_id', 'Unknown'),
                        'merops_id': row.get('merops_id', 'Unknown')
                    })
                    uniprot_to_merops[row.get('uniprot_id', 'Unknown')] = row.get('merops_id', 'Unknown')

        except Exception as e:
            print(f"Error reading CSV file {csv_file}: {e}")
            return

        print(f"Read {len(proteases)} proteases from {csv_file}")

        all_models = {}

        protease_stats = []

        for protease in proteases:
            uniprot_id = protease['uniprot_id']
            merops_id = protease['merops_id']
            gene_name = protease['gene_name']

            if merops_id == "Not found" or not merops_id:
                print(f"Skipping {uniprot_id} - no MEROPS ID")
                protease_stats.append({
                    'gene_name': gene_name,
                    'uniprot_id': uniprot_id,
                    'merops_id': merops_id,
                    'experimental_sites': 0,
                    'unique_substrates': 0,
                    'has_model': False,
                    'status': 'No MEROPS ID'
                })
                continue

            output_dir = self.set_protease_output_dir(uniprot_id)
            print(f"Processing {uniprot_id} ({merops_id}), output to {output_dir}")

            model, exp_count, unique_substrates = self.analyze_cleavage_sites(merops_id)

            status = 'Unknown'
            if model and model['frequency_matrix'] is not None:
                all_models[merops_id] = model
                status = 'Success'
                print(f"Successfully built prediction model for {merops_id}")
                has_model = True
            else:
                if exp_count == 0:
                    status = 'No experimental data'
                else:
                    status = 'Failed to predict true sites'
                print(f"Failed to build prediction model for {merops_id} - {status}")
                has_model = False
                self.close_connection()
                self.connect_to_db()

            protease_stats.append({
                'gene_name': gene_name,
                'uniprot_id': uniprot_id,
                'merops_id': merops_id,
                'experimental_sites': exp_count,
                'unique_substrates': unique_substrates,
                'has_model': has_model,
                'status': status
            })

        self.close_connection()

        stats_file = os.path.join(self.base_output_dir, "protease_site_statistics.csv")
        with open(stats_file, 'w', newline='') as f:
            fieldnames = ['gene_name', 'uniprot_id', 'merops_id', 'experimental_sites',
                        'unique_substrates', 'has_model', 'status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for stat in protease_stats:
                writer.writerow(stat)

        print(f"Saved protease statistics to {stats_file}")

        if all_models:
            self.save_models(all_models)

        if analyze_only:
            print("Analyze-only mode. Skipping prediction phase.")
            return

        if all_models:
            print(f"Successfully built {len(all_models)} prediction models. Processing FASTA files...")

            missing_peptides_file = os.path.join(self.base_output_dir, "missing_sequences_peptides.csv")
            missing_peptides = self.process_fasta_file("missing_sequences.fasta", all_models,
                                                    remove_n_term=20,
                                                    peptide_output_file=missing_peptides_file)

            processed_peptides_file = os.path.join(self.base_output_dir, "processed_by_SignalP6_peptides.csv")
            processed_peptides = self.process_fasta_file("processed_by_SignalP6.fasta", all_models,
                                                    remove_n_term=0,
                                                    peptide_output_file=processed_peptides_file)

            self.organize_results_by_protease(missing_peptides, processed_peptides, uniprot_to_merops)
        else:
            print("No valid prediction models were built. Skipping FASTA file processing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze proteases and predict cleavage sites')
    parser.add_argument('--analyze-only', action='store_true', help='Only run the analysis phase')
    parser.add_argument('--predict-only', action='store_true', help='Skip analysis and only run prediction')
    args = parser.parse_args()

    db_config = {
        'user': 'root',
        'database': 'merops',
        'unix_socket': '/var/run/mysqld/mysqld.sock'
    }

    analyzer = ProteaseAnalyzer(db_config)

    analyzer.load_local_proteome("uniprotkb_organism_id_9606_2025_05_13.fasta")

    analyzer.batch_process_proteases("merops_valid_id_mapping.csv",
                                  analyze_only=args.analyze_only,
                                  predict_only=args.predict_only)
