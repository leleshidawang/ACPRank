#!/usr/bin/env python3

import os
import pickle
import csv
import argparse
from Bio import SeqIO
from collections import defaultdict
import multiprocessing
import time
import sys
import gc

class MultiLevelPeptideGenerator:
    def __init__(self, base_dir=".", num_processes=6):
        self.base_dir = base_dir
        self.protease_analysis_dir = os.path.join(base_dir, "protease_analysis_batch")
        self.output_dir = os.path.join(base_dir, "multi_level_peptides")
        self.model_cache_file = os.path.join(self.protease_analysis_dir, "prediction_models.pkl")
        self.num_processes = num_processes

        self.LEVELS = [1]

        os.makedirs(self.output_dir, exist_ok=True)

        self.stats = {
            'total_sequences': 0,
            'total_proteases': 0,
            'total_peptides': 0,
            'start_time': time.time()
        }

    def load_models(self):
        if os.path.exists(self.model_cache_file):
            try:
                with open(self.model_cache_file, 'rb') as f:
                    all_models = pickle.load(f)
                print(f"Loaded {len(all_models)} prediction models")
                self.stats['total_proteases'] = len(all_models)
                return all_models
            except Exception as e:
                print(f"Error loading models: {e}")
                return None
        else:
            print(f"Model cache file {self.model_cache_file} not found")
            return None

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

    def generate_full_peptides(self, sequence, cleavage_sites, output_file):
        seq_len = len(sequence)
        N = len(cleavage_sites)

        if N == 0:
            return 0

        sorted_sites = sorted([site['position'] for site in cleavage_sites])

        with open(output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.get_field_names())

            uniprot_id = os.path.basename(os.path.dirname(output_file))
            protease_id = os.path.basename(output_file).split('_')[0]

            prev_pos = 0
            peptide_count = 0

            for pos in sorted_sites:
                if prev_pos < pos:
                    peptide = sequence[prev_pos:pos]
                    writer.writerow({
                        'substrate_uniprot': uniprot_id,
                        'protease_merops_id': protease_id,
                        'start_pos': prev_pos + 1,
                        'end_pos': pos,
                        'peptide': peptide,
                        'length': len(peptide),
                        'level': 'full',
                        'used_sites': N,
                        'start_score': self.get_site_score(cleavage_sites, prev_pos),
                        'end_score': self.get_site_score(cleavage_sites, pos-1)
                    })
                    peptide_count += 1
                prev_pos = pos

            if prev_pos < seq_len:
                peptide = sequence[prev_pos:seq_len]
                writer.writerow({
                    'substrate_uniprot': uniprot_id,
                    'protease_merops_id': protease_id,
                    'start_pos': prev_pos + 1,
                    'end_pos': seq_len,
                    'peptide': peptide,
                    'length': len(peptide),
                    'level': 'full',
                    'used_sites': N,
                    'start_score': self.get_site_score(cleavage_sites, prev_pos),
                    'end_score': 0
                })
                peptide_count += 1

            return peptide_count

    def generate_single_cut_peptides(self, sequence, cleavage_sites, output_file):
        seq_len = len(sequence)
        N = len(cleavage_sites)

        if N == 0:
            return 0

        sorted_sites = sorted([site['position'] for site in cleavage_sites])

        with open(output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.get_field_names())

            uniprot_id = os.path.basename(os.path.dirname(output_file))
            protease_id = os.path.basename(output_file).split('_')[0]

            peptide_count = 0

            for site in sorted_sites:
                if site > 0:
                    peptide = sequence[0:site]
                    writer.writerow({
                        'substrate_uniprot': uniprot_id,
                        'protease_merops_id': protease_id,
                        'start_pos': 1,
                        'end_pos': site,
                        'peptide': peptide,
                        'length': len(peptide),
                        'level': 'level_1',
                        'used_sites': 1,
                        'start_score': 0,
                        'end_score': self.get_site_score(cleavage_sites, site-1)
                    })
                    peptide_count += 1

                if site < seq_len:
                    peptide = sequence[site:seq_len]
                    writer.writerow({
                        'substrate_uniprot': uniprot_id,
                        'protease_merops_id': protease_id,
                        'start_pos': site + 1,
                        'end_pos': seq_len,
                        'peptide': peptide,
                        'length': len(peptide),
                        'level': 'level_1',
                        'used_sites': 1,
                        'start_score': self.get_site_score(cleavage_sites, site),
                        'end_score': 0
                    })
                    peptide_count += 1

            return peptide_count

    def get_field_names(self):
        return ['substrate_uniprot', 'protease_merops_id', 'start_pos',
                'end_pos', 'peptide', 'length', 'start_score',
                'end_score', 'level', 'used_sites']

    def get_site_score(self, cleavage_sites, position):
        for site in cleavage_sites:
            if site['position'] == position:
                return site['score']
        return 0

    def process_sequence_model(self, args):
        uniprot_id, sequence, model_id, model, output_dir = args
        os.makedirs(output_dir, exist_ok=True)

        print(f"Process {os.getpid()} predicting cleavage sites for {uniprot_id} using protease {model_id}")

        try:
            frequency_matrix = model['frequency_matrix']
            position_weights = model['position_weights']
            threshold = model['min_tp_score'] if 'min_tp_score' in model and model['min_tp_score'] is not None else 0.05

            predictions = self.predict_cleavage_sites(
                sequence, frequency_matrix, position_weights, threshold
            )

            if not predictions:
                print(f"  No cleavage sites found for {uniprot_id} using protease {model_id}")
                return 0

            n_sites = len(predictions)
            print(f"  Found {n_sites} cleavage sites using protease {model_id}")

            output_file = os.path.join(output_dir, f"{model_id}_multi_level_peptides.csv")

            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.get_field_names())
                writer.writeheader()

            full_count = self.generate_full_peptides(sequence, predictions, output_file)
            print(f"  Generated {full_count} full cleavage peptides for {uniprot_id} using protease {model_id}")

            single_count = self.generate_single_cut_peptides(sequence, predictions, output_file)
            print(f"  Generated {single_count} single cut peptides for {uniprot_id} using protease {model_id}")

            total_peptides = full_count + single_count

            gc.collect()

            elapsed_time = time.time() - self.stats['start_time']
            print(f"  Total {total_peptides} peptides generated for {uniprot_id} using protease {model_id} and saved to {output_file}")
            print(f"  Elapsed time: {elapsed_time:.2f} seconds")

            return total_peptides

        except Exception as e:
            print(f"Error processing {uniprot_id} using protease {model_id}: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def process_fasta_files(self):
        self.stats['start_time'] = time.time()

        models = self.load_models()
        if not models:
            print("Failed to load prediction models, exiting")
            return

        uniprot_to_merops = self.load_uniprot_to_merops_mapping()

        processed_sequences = self.extract_peptides_from_fasta(
            os.path.join(self.base_dir, "processed_by_SignalP6.fasta")
        )

        missing_sequences = self.extract_peptides_from_fasta(
            os.path.join(self.base_dir, "missing_sequences.fasta"),
            remove_n_term=20
        )

        all_sequences = {**processed_sequences, **missing_sequences}
        self.stats['total_sequences'] = len(all_sequences)

        pool = multiprocessing.Pool(processes=self.num_processes)

        tasks = []
        for uniprot_id, seq_info in all_sequences.items():
            sequence = seq_info['sequence']
            uniprot_output_dir = os.path.join(self.output_dir, uniprot_id)

            for model_id, model in models.items():
                if not model or not model['frequency_matrix']:
                    continue

                tasks.append((uniprot_id, sequence, model_id, model, uniprot_output_dir))

        print(f"Starting parallel processing of {len(tasks)} tasks using {self.num_processes} processes")

        results = []
        for i, result in enumerate(pool.imap_unordered(self.process_sequence_model, tasks)):
            results.append(result)
            if (i+1) % 100 == 0 or (i+1) == len(tasks):
                print(f"Completed {i+1}/{len(tasks)} tasks ({(i+1)/len(tasks)*100:.1f}%)")
                print(f"Current peptides generated: {sum(results):,}")
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_usage = process.memory_info().rss / 1024 / 1024
                    print(f"Current memory usage: {memory_usage:.1f} MB")
                except:
                    pass

        pool.close()
        pool.join()

        total_peptides = sum(results)
        total_time = time.time() - self.stats['start_time']

        print(f"\n====== Processing Complete ======")
        print(f"Total sequences: {self.stats['total_sequences']}")
        print(f"Total proteases: {self.stats['total_proteases']}")
        print(f"Total peptides: {total_peptides:,}")
        print(f"Total running time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        print(f"Average peptides per sequence-protease combination: {total_peptides/len(tasks):.2f}")
        print(f"All peptides saved to {self.output_dir} directory")

    def load_uniprot_to_merops_mapping(self):
        mapping = {}
        csv_file = os.path.join(self.base_dir, "merops_valid_id_mapping.csv")

        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uniprot_id = row.get('uniprot_id', 'Unknown')
                    merops_id = row.get('merops_id', 'Unknown')
                    if merops_id != 'Not found' and merops_id:
                        mapping[uniprot_id] = merops_id

            print(f"Loaded {len(mapping)} UniProt to MEROPS mappings")
            return mapping
        except Exception as e:
            print(f"Error loading mappings: {e}")
            return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate multi-level peptide cleavage results')
    parser.add_argument('--processes', type=int, default=6, help='Number of CPU processes to use')
    args = parser.parse_args()

    generator = MultiLevelPeptideGenerator(num_processes=args.processes)
    generator.process_fasta_files()
