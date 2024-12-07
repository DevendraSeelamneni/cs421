import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from src.embeddings_extraction import TargetEmbeddingsExtraction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Embeddings Extraction for WiC pairs', add_help=True)
    parser.add_argument('-d', '--dir',
                        type=str,
                        help='Directory containing WiC datasets processed')
    parser.add_argument('-m', '--model',
                        type=str,
                        default='bert-base-uncased',
                        help='Pre-trained bert-like model')
    parser.add_argument('-s', '--subword_prefix',
                        type=str,
                        default='##',
                        help='Subword prefix')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=16,
                        help='Batch size')
    parser.add_argument('-M', '--max_length',
                        type=int,
                        default=None,
                        help='Max length used for tokenization')
    parser.add_argument('-g', '--use_gpu',
                        action='store_true',
                        help='If true, use GPU for embeddings extraction')
    parser.add_argument('-T', '--train_set',
                        action='store_true',
                        help='If true, extract embeddings for the train set')
    parser.add_argument('-t', '--test_set',
                        action='store_true',
                        help='If true, extract embeddings for the test set')
    parser.add_argument('-D', '--dev_set',
                        action='store_true',
                        help='If true, extract embeddings for the dev set')
    parser.add_argument('--max_examples',
                        type=int,
                        default=None,
                        help='Maximum number of examples to process per dataset')
    args = parser.parse_args()

    # Create extractor
    extractor = TargetEmbeddingsExtraction(args.model, subword_prefix=args.subword_prefix, use_gpu=args.use_gpu)
    extractor.add_token_to_vocab()  # Add token [RANDOM]

    # Datasets to process
    sets = []
    if args.dev_set:
        sets.append('dev')
    if args.train_set:
        sets.append('train')
    if args.test_set:
        sets.append('test')

    bar = tqdm(sets, total=len(sets))
    for s in bar:
        bar.set_description(s)

        # Create output directories
        Path(f'{args.dir}/target_embeddings/{args.model.replace("/", "_")}/{s}').mkdir(parents=True, exist_ok=True)

        input_filename = f'{args.dir}/{s}.txt'

        # Limit the number of examples if --max_examples is specified
        if args.max_examples:
            with open(input_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            lines = lines[:args.max_examples]
            
            # Save the truncated dataset temporarily
            temp_filename = f'{input_filename}.temp'
            with open(temp_filename, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            input_filename = temp_filename

        # Extraction
        embeddings = extractor.extract_embeddings(dataset=input_filename,
                                                  batch_size=args.batch_size,
                                                  max_length=args.max_length)

        # Remove the temporary file if created
        if args.max_examples:
            Path(temp_filename).unlink()

        # Store embeddings for each layer
        for layer in embeddings.keys():
            output_filename = f'{args.dir}/target_embeddings/{args.model.replace("/", "_")}/{s}/{layer}.pt'
            torch.save(embeddings[layer].to('cpu'), output_filename)

        # Update bar
        bar.update(1)
