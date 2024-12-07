import json
from pathlib import Path

# Base directory for your dataset
base_dir = '/content/project_folder/asc-lr-main/WiC/mclwic_en/MCL-WiC/dev/multilingual'

for lang, sets in zip(['en', 'fr'], [['dev', 'train', 'test'], ['dev', 'test']]):
    for s in sets:
        # Adjust paths to match your structure
        data_path = f'{base_dir}/{s}.{lang}-{lang}.data'
        gold_path = f'{base_dir}/{s}.{lang}-{lang}.gold'

        # Check if files exist before proceeding
        if not Path(data_path).exists() or not Path(gold_path).exists():
            print(f"Files not found: {data_path} or {gold_path}. Skipping...")
            continue

        data = json.load(open(data_path, mode='r', encoding='utf-8'))
        gold = json.load(open(gold_path, mode='r', encoding='utf-8'))

        records = list()

        for i, pair in enumerate(data):
            for j in range(1, 3):
                record = dict()
                record['lemma'] = pair['lemma']

                # Map POS tags
                if pair['pos'] == 'NOUN':
                    record['pos'] = 'N'
                elif pair['pos'] == 'VERB':
                    record['pos'] = 'V'
                elif pair['pos'] == 'ADJ':
                    record['pos'] = 'A'
                elif pair['pos'] == 'ADV':
                    record['pos'] = 'R'

                record['token'] = pair[f'sentence{j}'][int(pair[f'start{j}']):int(pair[f'end{j}'])]
                record['start'] = int(pair[f'start{j}'])
                record['end'] = int(pair[f'end{j}'])
                record['sentence'] = pair[f'sentence{j}']
                record['gold'] = int(gold[i]['tag'] == 'T')
                records.append(json.dumps(record) + '\n')

        # Create output directory and write processed file
        Path(f'mclwic_{lang}').mkdir(parents=True, exist_ok=True)
        with open(f'mclwic_{lang}/{s}.txt', mode='w', encoding='utf-8') as f:
            f.writelines(records)
