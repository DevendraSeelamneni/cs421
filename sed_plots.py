import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Configure Matplotlib font size
import matplotlib
matplotlib.rcParams.update({'font.size': 13})

def plot_bert_metrics(model_path: str, layers: list, filenames: list, pos_tags: list):
    '''Plot metrics for BERT.'''
    colors = {
        'antonyms': '#D55E00',
        'synonyms': '#56B4E9',
        'hypernyms': '#009E73',
        'random': '#000000',
    }

    fig, axs = plt.subplots(1, len(pos_tags), figsize=(15, 5))

    for i, pos in enumerate(pos_tags):
        for filename in filenames:
            values = []
            for layer in layers:
                # Define the file path
                path = f'{model_path}/{layer}/{filename}_{pos}.pkl'
                if not os.path.exists(path):
                    continue  # Skip missing files

                # Load data from the pickle file
                with open(path, 'rb') as f:
                    data = pickle.load(f)

                # Flatten and compute mean while handling variable-length arrays
                if isinstance(data, list):
                    data = np.concatenate([np.asarray(d).flatten() for d in data])
                elif isinstance(data, np.ndarray):
                    data = data.flatten()

                if len(data) > 0:  # Ensure non-empty data
                    values.append(np.mean(data))

            # Plot the data if values are available
            if len(values) > 0:
                axs[i].plot(
                    layers[:len(values)],  # Only plot available layers
                    values,
                    label=filename,
                    color=colors.get(filename, '#000000'),
                    linewidth=1.5,
                    marker='o'
                )

        axs[i].set_title(f'POS: {pos.capitalize()}')
        axs[i].set_xlabel('Layers')
        axs[i].set_ylabel('Self-embedding Distance (SED)')
        axs[i].legend()

    plt.tight_layout()
    output_path = os.path.join(model_path, 'bert_sed_plot.png')
    plt.savefig(output_path)
    plt.show()
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    # Define default parameters
    model_path = '/content/project_folder/asc-lr-main/bert/cosine_distances'
    layers = [1, 2, 3, 4]  # Default to 4 layers
    filenames = ['antonyms', 'synonyms', 'hypernyms', 'random']
    pos_tags = ['v', 'r', 'a', 'n']  # POS: verb, adverb, adjective, noun

    # Generate the plot
    plot_bert_metrics(model_path, layers, filenames, pos_tags)
