import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt



if __name__ == '__main__':
    # Load the classification reports from the json files
    # cmp_methods = ['cube-10x10x10', 'cube-20x20x20']
    
    output_dir = 'stat_results'
    os.makedirs(output_dir, exist_ok=True)
    
    cmp_methods = ['our_method', 'baseline']
    
    
    
    # classification_reports = {
    #     'subject1': {cmp_methods[0]: pd.read_json('result-rf-cnn-yeo-RandomSearch/subj1/results.json').iloc[-1]['Classification report'],
    #                  cmp_methods[1]: pd.read_json('result-baseline/icassp_haxby2001_subj1_vt_cnn-1/results.json').iloc[-1]['Classification report']},
        
    #     'subject2': {cmp_methods[0]: pd.read_json('result-rf-cnn-yeo-RandomSearch/subj2/results.json').iloc[-1]['Classification report'],
    #                  cmp_methods[1]: pd.read_json('result-baseline/icassp_haxby2001_subj2_vt_cnn-1/results.json').iloc[-1]['Classification report']},
        
    #     'subject3': {cmp_methods[0]: pd.read_json('result-rf-cnn-yeo-RandomSearch/subj3/results.json').iloc[-1]['Classification report'],
    #                  cmp_methods[1]: pd.read_json('result-baseline/icassp_haxby2001_subj3_vt_cnn-1/results.json').iloc[-1]['Classification report']},
        
    #     'subject4': {cmp_methods[0]: pd.read_json('result-rf-cnn-yeo-RandomSearch/subj4/results.json').iloc[-1]['Classification report'],
    #                  cmp_methods[1]: pd.read_json('result-baseline/icassp_haxby2001_subj4_vt_cnn-1/results.json').iloc[-1]['Classification report']},
        
    #     'subject5': {cmp_methods[0]: pd.read_json('result-rf-cnn-yeo-RandomSearch/subj5/results.json').iloc[-1]['Classification report'],
    #                  cmp_methods[1]: pd.read_json('result-baseline/icassp_haxby2001_subj5_vt_cnn-1/results.json').iloc[-1]['Classification report']},

    # }
    
    
    our_method = [pd.read_json(f'result-rf-svm-yeo-RandomSearch/subj{i}/results.json').iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)]
    baseline_method = [pd.read_json(f'result-baseline/icassp_haxby2001_subj{i}_vt_svm-1/results.json').iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)]
    


    # Choose your performance metric
    # metric = 'f1-score'
    metric = 'accuracy'

    ################## Perform the t-test for the weighted average ##################
    # Collect the metric for each method and subject
    # our_method = [report[cmp_methods[0]]['weighted avg'][metric] for report in classification_reports.values()]
    # baseline_method = [report[cmp_methods[1]]['weighted avg'][metric] for report in classification_reports.values()]
    
    # Print classification reports
    # print("Our method")
    # for report in classification_reports.values():
    #     print(report[cmp_methods[0]]['weighted avg'])
    # print("Baseline method")
    # for report in classification_reports.values():
    #     print(report[cmp_methods[1]]['weighted avg'])

    # Perform the t-test
    t, p = stats.ttest_rel(our_method, baseline_method)
    
    ################## Plot the results for weighted average ##################

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

    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/weighted_avg.png')
    # Clearing the plot
    plt.clf()
    
    
    
    ################## Perform the t-test for each class ##################
    # Get your classes
    classes = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']

    for class_name in classes:
        print(f"Results for class: {class_name}")
        
        # Collect the metric for each method and subject
        # our_method_class = [report[cmp_methods[0]][class_name][metric] for report in classification_reports.values()]
        # baseline_method_class = [report[cmp_methods[1]][class_name][metric] for report in classification_reports.values()]
        
        our_method_class = [pd.read_json(f'result-rf-cnn-yeo-RandomSearch/subj{i}/results.json').iloc[-1]['Classification report'][class_name][metric] for i in range(1, 6)]
        baseline_method_class = [pd.read_json(f'result-baseline/icassp_haxby2001_subj{i}_vt_cnn-1/results.json').iloc[-1]['Classification report'][class_name][metric] for i in range(1, 6)]
        
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
        

        plt.tight_layout()
        # Display the plot
        plt.savefig(f'{output_dir}/{class_name}.png')
        plt.clf()
     
     
     