"""
Perform t-test to see which model is the best model
"""

import pandas as pd
from scipy import stats

def generate_t_test(output1, output2):
    """
    Compare two outputs from two different models using the paired t-test formula.
    Return t_statistic and p_value.

    Parameters:
    output1 (list): List of scores for the first model.
    output2 (list): List of scores for the second model.

    Returns:
    tuple: t_statistic and p_value from the t-test.
    """
    t_statistic, p_value = stats.ttest_rel(output1, output2, alternative='two-sided')
    return t_statistic, p_value

def run_t_test(baseline, model_1, model_2, measure_name):
    """
    Run paired t-tests to compare the performance of three models.

    Parameters:
    baseline (list): List of scores for the baseline model (BM25).
    model_1 (list): List of scores for the first model (JM_LM).
    model_2 (list): List of scores for the second model (My_PRM).
    measure_name (str): Name of the performance measure being compared.

    Returns:
    None
    """
    # Print t-test results for all comparisons
    print(f"\nComparing BM25 and My_PRM models for {measure_name}")
    bm25_my_prm = generate_t_test(baseline, model_2)
    print("T-Statistic:", bm25_my_prm[0])
    print("P-Value:", bm25_my_prm[1])

    print(f"\nComparing JM_LM and My_PRM models for {measure_name}")
    jm_lm_my_prm = generate_t_test(model_1, model_2)
    print("T-Statistic:", jm_lm_my_prm[0])
    print("P-Value:", jm_lm_my_prm[1])

    print(f"\nComparing BM25 and JM_LM models for {measure_name}")
    bm25_jm_lm = generate_t_test(baseline, model_1)
    print("T-Statistic:", bm25_jm_lm[0])
    print("P-Value:", bm25_jm_lm[1])

    # Analyze the results and recommend the best model
    if bm25_my_prm[1] < 0.05:
        print(f"\nRecommendation for {measure_name}: My_PRM model is significantly better than BM25.")
    if jm_lm_my_prm[1] < 0.05:
        print(f"\nRecommendation for {measure_name}: My_PRM model is significantly better than JM_LM.")
    if bm25_my_prm[1] >= 0.05 and jm_lm_my_prm[1] >= 0.05 and bm25_jm_lm[1] >= 0.05:
        print(f"\nRecommendation for {measure_name}: No significant difference found. Further analysis needed.")

def load_and_run_tests(ap_file, dcg_file, p10_file):
    """
    Load evaluation results from CSV files and run t-tests.

    Parameters:
    ap_file (str): Path to the CSV file containing Average Precision (AP) scores.
    dcg_file (str): Path to the CSV file containing DCG@10 scores.
    p10_file (str): Path to the CSV file containing P@10 scores.

    Returns:
    None
    """
    try:
        # Load the evaluation results from CSV files
        ap_df = pd.read_csv(ap_file)
        dcg_df = pd.read_csv(dcg_file)
        p10_df = pd.read_csv(p10_file)

        # Ensure the first 50 rows are relevant
        ap_baseline = ap_df['BM25'][:50].tolist()
        ap_model_1 = ap_df['JM_LM'][:50].tolist()
        ap_model_2 = ap_df['My_PRM'][:50].tolist()

        dcg_baseline = dcg_df['BM25'][:50].tolist()
        dcg_model_1 = dcg_df['JM_LM'][:50].tolist()
        dcg_model_2 = dcg_df['My_PRM'][:50].tolist()

        p10_baseline = p10_df['BM25'][:50].tolist()
        p10_model_1 = p10_df['JM_LM'][:50].tolist()
        p10_model_2 = p10_df['My_PRM'][:50].tolist()

        # Run t-tests for AP
        run_t_test(ap_baseline, ap_model_1, ap_model_2, "Average Precision (AP)")

        # Run t-tests for DCG@10
        run_t_test(dcg_baseline, dcg_model_1, dcg_model_2, "DCG@10")

        # Run t-tests for P@10
        run_t_test(p10_baseline, p10_model_1, p10_model_2, "P@10")

    except Exception as e:
        print(f"Error during loading or processing files: {e}")

# File paths for the evaluation result CSV files
ap_file = 'evaluation_results_ap.csv'
dcg_file = 'evaluation_results_dcg10.csv'
p10_file = 'evaluation_results_p10.csv'

# Run the tests and get recommendations
load_and_run_tests(ap_file, dcg_file, p10_file)
