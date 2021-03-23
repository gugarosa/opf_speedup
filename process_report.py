import argparse
import pickle

import opfython.math.distance as d


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Processes the report into a text file.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['arcene', 'basehock', 'caltech101', 'coil20',
                                                                       'isolet', 'lung', 'madelon', 'mpeg7', 'mpeg7_BAS',
                                                                       'mpeg7_FOURIER', 'mushrooms', 'ntl-commercial',
                                                                       'ntl-industrial', 'orl', 'pcmac', 'phishing',
                                                                       'segment', 'semeion', 'sonar', 'spambase',
                                                                       'vehicle', 'wine'])

    parser.add_argument('seed', help='Deterministic seed', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    seed = args.seed
    input_file = f'outputs/{dataset}_{seed}_report.pkl'
    output_file = f'outputs/{dataset}_{seed}_report.txt'

    # Opening the input file
    with open(input_file, 'rb') as f:
        # Loading the reports
        report = pickle.load(f)

    # Opening the output file
    with open(output_file, 'w') as f:
        # Writing the header
        f.write('distance,accuracy,precision,recall,f1\n')

        # Iterates through every distance metric
        for i, key in enumerate(d.DISTANCES.keys()):
            # Writing the metrics
            f.write(f"{key},{report[i]['accuracy']},{report[i]['macro avg']['precision']},"
                    f"{report[i]['macro avg']['recall']},{report[i]['macro avg']['f1-score']}\n")
