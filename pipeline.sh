# Datasets
DATASETS=("arcene" "basehock" "caltech101" "coil20"
          "isolet" "lung" "madelon" "mpeg7" "mpeg7_BAS"
          "mpeg7_FOURIER" "mushrooms" "ntl-commercial"
          "ntl-industrial" "orl" "pcmac" "phishing"
          "segment" "semeion" "sonar" "spambase"
          "vehicle" "wine")

# Percentage of training set
SPLIT=0.25

# Number of runnings
N_RUNS=25

# For every running
for RUN in $(seq 1 $N_RUNS); do
    # For every dataset
    for DATA in "${DATASETS[@]}"; do
        # Benchmarks the time
        python benchmark_time.py $DATA -tr_split $SPLIT -seed $RUN --normalize

        # Process the report
        python process_report.py $DATA $RUN
    done
done
