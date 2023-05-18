import pandas as pd
from scipy import stats


if __name__ == '__main__':
    # Load the classification reports from the CSV files
    classification_reports = {
        # 'subject1': {'our_method': pd.read_json('results/icassp_haxby2001_subj1_rf_curf/results.json').iloc[-1]['Classification report'],
        #              'baseline_method': pd.read_json('results/icassp_haxby2001_subj1_vt_rf/results.json').iloc[0]['Classification report']},
        # 'subject2': {'our_method': pd.read_json('results/icassp_haxby2001_subj2_rf_curf/results.json').iloc[-1]['Classification report'],
        #                 'baseline_method': pd.read_json('results/icassp_haxby2001_subj2_vt_rf/results.json').iloc[0]['Classification report']},
        'subject3': {'our_method': pd.read_json('results/icassp_haxby2001_subj3_rf_curf/results.json').iloc[-1]['Classification report'],
                        'baseline_method': pd.read_json('results/icassp_haxby2001_subj3_vt_rf/results.json').iloc[0]['Classification report']},
        'subject4': {'our_method': pd.read_json('results/icassp_haxby2001_subj4_rf_curf/results.json').iloc[-1]['Classification report'],
                        'baseline_method': pd.read_json('results/icassp_haxby2001_subj4_vt_rf/results.json').iloc[0]['Classification report']},
        'subject5': {'our_method': pd.read_json('results/icassp_haxby2001_subj5_rf_curf/results.json').iloc[-1]['Classification report'],
                        'baseline_method': pd.read_json('results/icassp_haxby2001_subj5_vt_rf/results.json').iloc[0]['Classification report']}
   
    }

    # Choose your performance metric
    metric = 'f1-score'

    ################## Perform the t-test for the weighted average ##################
    # Collect the metric for each method and subject
    our_method = [report['our_method']['weighted avg'][metric] for report in classification_reports.values()]
    baseline_method = [report['baseline_method']['weighted avg'][metric] for report in classification_reports.values()]

    # Perform the t-test
    t, p = stats.ttest_rel(our_method, baseline_method)
    
    print(f'The t-statistic is {t} and the p-value is {p}')
    
    ################## Perform the t-test for each class ##################
    # Get your classes
    classes = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']

    for class_name in classes:
        print(f"Results for class: {class_name}")
        
        # Collect the metric for each method and subject
        our_method_class = [report['our_method'][class_name][metric] for report in classification_reports.values()]
        baseline_method_class = [report['baseline_method'][class_name][metric] for report in classification_reports.values()]
        
        # Perform the t-test
        t, p = stats.ttest_rel(our_method_class, baseline_method_class)
        
        print(f"t-statistic: {t}, p-value: {p}\n")
