import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.distance import cosine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Computing cosine distances between word and context', add_help=True)
    parser.add_argument('-e', '--embeddings_dir',
                        type=str,
                        help='Directory containing Lexical Replacement datasets')
    parser.add_argument('-m', '--model_type',
                        type=str,
                        default='bert',
                        help='Type of the model used')
    parser.add_argument('-l', '--layer',
                        default=12,
                        type=int,
                        help='Embedding layer')
    parser.add_argument('-t', '--target_index_dir',
                        type=str,
                        help='Directory containing indexes of target words/replacements')
    parser.add_argument('-s', '--special_tokens_mask_dir',
                        type=str,
                        help='Directory containing special token mask indexes')
    args = parser.parse_args()

    model_type = args.model_type
    
    # create CD dir
    Path(f'{model_type}/distance_context_original_context_replacement/{args.layer}').mkdir(parents=True, exist_ok=True)
    Path(f'{model_type}/distance_context_original_original/{args.layer}').mkdir(parents=True, exist_ok=True)
    Path(f'{model_type}/distance_context_replacement_replacement/{args.layer}').mkdir(parents=True, exist_ok=True)
    Path(f'{model_type}/distance_context_original_replacement/{args.layer}').mkdir(parents=True, exist_ok=True)
    Path(f'{model_type}/distance_context_replacement_original/{args.layer}').mkdir(parents=True, exist_ok=True)

    paths = list(Path(f'{args.embeddings_dir}/{args.layer}/').glob("*.pt"))
    bar = tqdm(paths, total=len(paths))
    for rep_file in bar:
        # files and filenames
        rep_file_pt = str(rep_file)
        filename = os.path.basename(rep_file)[:-3]
        rep_idx_file_npy = f'{args.target_index_dir}/{filename}.npy'
        rep_special_tokens_mask_pt = f'{args.special_tokens_mask_dir}/{filename}.pt'

        # update bar description
        bar.set_description(filename)

        # load embeddings, indexes, special_token_masks
        X = torch.load(rep_file_pt)
        idx = np.load(rep_idx_file_npy)
        special_tokens_mask = torch.load(rep_special_tokens_mask_pt)

        # wrapper cd values
        cd_context_original_context_replacement = list()
        cd_context_original_original = list()
        cd_context_replacement_replacement = list()
        cd_context_replacement_original = list()
        cd_context_original_replacement = list()

        for i in range(0, X.shape[0], 2):
            # indexes of the target word/replacement
            idx_replacement = [int(token) for token in idx[i].split()]
            idx_original = [int(token) for token in idx[i + 1].split()]

            # special token masks for the original and synthetic sentence
            special_token_mask_replacement = special_tokens_mask[i]
            special_token_mask_original = special_tokens_mask[i + 1]

            model_length = 512

            # context: ASSUMPTION - NO [UNK] tokens otherwise for correct alignments
            # indexes of context words for the original and synthetic sentence
            context_index_replacement = 1 - special_token_mask_replacement
            context_index_original = 1 - special_token_mask_original
            context_index_replacement[idx_replacement] = False
            context_index_original[idx_original] = False

            # word/replacement
            index_replacement = torch.zeros(model_length, dtype=bool)
            index_original = torch.zeros(model_length, dtype=bool)
            index_replacement[idx_replacement] = True
            index_original[idx_original] = True

            # First sub-token of the target
            position_word = idx_replacement[0]  # equivalent to (index_original==True).nonzero().squeeze()[0]

            # Split embeddings
            # context embeddings
            X_context_replacement = X[i][context_index_replacement == 1]
            X_context_original = X[i + 1][context_index_original == 1]

            # target embedding (mean if sub-tokens)
            X_replacement = X[i][index_replacement].mean(axis=0)
            X_original = X[i + 1][index_original].mean(axis=0)

            # print(X_replacement.shape, X_original.shape)
            # print(X_context_replacement.shape, X_context_original.shape)

            # (position word-1) -> cls token removed
            new_position_word = position_word - 1

            # join embeddings
            processed_X_replacement = torch.vstack(
                [X_context_replacement[:new_position_word], X_context_replacement[new_position_word + 1:]])
            processed_X_original = torch.vstack(
                [X_context_original[:new_position_word], X_context_original[new_position_word + 1:]])

            # pairwise cosine distance
            cd_context_original_context_replacement.append(
                cosine(processed_X_replacement.mean(axis=0), processed_X_original.mean(axis=0)))
            cd_context_original_original.append(cosine(X_original, processed_X_original.mean(axis=0)))
            cd_context_replacement_replacement.append(cosine(X_replacement, processed_X_replacement.mean(axis=0)))
            cd_context_replacement_original.append(cosine(X_original, processed_X_replacement.mean(axis=0)))
            cd_context_original_replacement.append(cosine(X_replacement, processed_X_original.mean(axis=0)))

        cd_context_original_context_replacement_filename = f'{model_type}/distance_context_original_context_replacement/{args.layer}/{filename}.npy'
        np.save(cd_context_original_context_replacement_filename, np.array(cd_context_original_context_replacement))
        cd_context_original_original_filename = f'{model_type}/distance_context_original_original/{args.layer}/{filename}.npy'
        np.save(cd_context_original_original_filename, np.array(cd_context_original_original))
        cd_context_replacement_replacement_filename = f'{model_type}/distance_context_replacement_replacement/{args.layer}/{filename}.npy'
        np.save(cd_context_replacement_replacement_filename, np.array(cd_context_replacement_replacement))
        cd_context_replacement_original_filename = f'{model_type}/distance_context_replacement_original/{args.layer}/{filename}.npy'
        np.save(cd_context_replacement_original_filename, np.array(cd_context_replacement_original))
        cd_context_original_replacement_filename = f'{model_type}/distance_context_original_replacement/{args.layer}/{filename}.npy'
        np.save(cd_context_original_replacement_filename, np.array(cd_context_original_replacement))

        bar.update(1)
