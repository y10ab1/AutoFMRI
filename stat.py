import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



if __name__ == '__main__':
    # Load the classification reports from the json files
    cmp_methods = ['cube-10x10x10', 'cube-20x20x20']
    classification_reports = {
        'subject1': {cmp_methods[0]: pd.read_json('result-10x10x10-40patches-subj1/results.json').iloc[-1]['Classification report'],
                     cmp_methods[1]: pd.read_json('icassp-results/icassp_haxby2001_subj1_rf_curf/results.json').iloc[-1]['Classification report']},
        'subject2': {cmp_methods[0]: pd.read_json('result-10x10x10-40patches-subj2/results.json').iloc[-1]['Classification report'],
                        cmp_methods[1]: pd.read_json('icassp-results/icassp_haxby2001_subj2_rf_curf/results.json').iloc[-1]['Classification report']},
        'subject3': {cmp_methods[0]: pd.read_json('result-10x10x10-40patches-subj3/results.json').iloc[-1]['Classification report'],
                        cmp_methods[1]: pd.read_json('icassp-results/icassp_haxby2001_subj3_rf_curf/results.json').iloc[-1]['Classification report']},
        'subject4': {cmp_methods[0]: pd.read_json('result-10x10x10-40patches-subj4/results.json').iloc[-1]['Classification report'],
                        cmp_methods[1]: pd.read_json('icassp-results/icassp_haxby2001_subj4_rf_curf/results.json').iloc[-1]['Classification report']},
        'subject5': {cmp_methods[0]: pd.read_json('result-10x10x10-40patches-subj5/results.json').iloc[-1]['Classification report'],
                        cmp_methods[1]: pd.read_json('icassp-results/icassp_haxby2001_subj5_rf_curf/results.json').iloc[-1]['Classification report']},

    }
        

    # Choose your performance metric
    metric = 'f1-score'

    ################## Perform the t-test for the weighted average ##################
    # Collect the metric for each method and subject
    our_method = [report[cmp_methods[0]]['weighted avg'][metric] for report in classification_reports.values()]
    baseline_method = [report[cmp_methods[1]]['weighted avg'][metric] for report in classification_reports.values()]

    # Perform the t-test
    t, p = stats.ttest_rel(our_method, baseline_method)
    
    ################## Plot the results for weighted average ##################
    # Significance levels for different levels of statistical significance
    significance_levels = {
        0.001: '***',
        0.01: '**',
        0.05: '*'
    }

    # Significance level to be used for visualization
    visualization_significance = 0.05

    # Perform the t-test for the weighted average
    # (Assuming you have already calculated 't' and 'p' for the weighted average)
    print("Results for the weighted average")
    print(f"The t-statistic is {t} and the p-value is {p}\n")

    # Plotting the bar plot for the weighted average
    plt.bar([0, 1], [np.mean(our_method), np.mean(baseline_method)], yerr=[np.std(our_method), np.std(baseline_method)],
            tick_label=cmp_methods)

    # Adding labels and title
    plt.xlabel('Methods')
    plt.ylabel(metric)
    plt.title(f'Comparison of {cmp_methods[0]} vs. {cmp_methods[1]} (Weighted Average)')

#     # Adding significance indicators
#     for significance_level, significance_label in significance_levels.items():
#         if p < significance_level:
#             plt.text(0.5, np.mean(our_method) + np.std(our_method) + 0.01, significance_label,
#                     ha='center', fontsize=12)
#             break
    
    plt.tight_layout()
    plt.savefig('icassp-results/weighted_avg.png')
    # Clearing the plot
    plt.clf()
    
    
    
    ################## Perform the t-test for each class ##################
    # Get your classes
    classes = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']

    for class_name in classes:
        print(f"Results for class: {class_name}")
        
        # Collect the metric for each method and subject
        our_method_class = [report[cmp_methods[0]][class_name][metric] for report in classification_reports.values()]
        baseline_method_class = [report[cmp_methods[1]][class_name][metric] for report in classification_reports.values()]
        
        # Perform the t-test
        t, p = stats.ttest_rel(our_method_class, baseline_method_class)
        
        print(f"t-statistic: {t}, p-value: {p}\n")

        ################## Plot the results for each class ##################
        # Plotting the bar plot for each class
        plt.bar([0, 1], [np.mean(our_method_class), np.mean(baseline_method_class)],
                yerr=[np.std(our_method_class), np.std(baseline_method_class)], tick_label=cmp_methods)

        # Adding labels and title
        plt.xlabel('Methods')
        plt.ylabel(metric)
        plt.title(f'Comparison of {cmp_methods[0]} vs. {cmp_methods[1]} ({class_name})')

        # # Adding significance indicators
        # for significance_level, significance_label in significance_levels.items():
        #     if p < significance_level:
        #         plt.text(0.5, np.mean(our_method) + np.std(our_method) + 0.01, significance_label,
        #                 ha='center', fontsize=12)
        #         break
        

        plt.tight_layout()
        # Display the plot
        plt.savefig(f'icassp-results/{class_name}.png')
        plt.clf()
     
     