import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns




if __name__ == '__main__':

    
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
    
    # Example data in a dictionary format
    data = {
        'Subject': ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'Subject 5'],
        'ROI + RF': [(acc := pd.read_json(f'result-baseline/icassp_haxby2001_subj{i}_vt_rf-1/results.json')).iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)],
        'ROI + SVM': [(acc := pd.read_json(f'result-baseline/icassp_haxby2001_subj{i}_vt_svm-1/results.json')).iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)],
        'ROI + CNN': [(acc := pd.read_json(f'result-baseline/icassp_haxby2001_subj{i}_vt_cnn-1/results.json')).iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)],
        'ROI + ANOVA + RF': [(acc := pd.read_json(f'result-baseline-anova/icassp_haxby2001_subj{i}_vt_rf-1/results.json')).iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)],
        'ROI + ANOVA + SVM': [(acc := pd.read_json(f'result-baseline-anova/icassp_haxby2001_subj{i}_vt_svm-1/results.json')).iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)],
        'ROI + ANOVA + CNN': [(acc := pd.read_json(f'result-baseline-anova/icassp_haxby2001_subj{i}_vt_cnn-1/results.json')).iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)],
        'ROI + RFE + RF': [(acc := pd.read_json(f'result-baseline-rfe/icassp_haxby2001_subj{i}_vt_rf-1/results.json')).iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)],
        'ROI + RFE + SVM': [(acc := pd.read_json(f'result-baseline-rfe/icassp_haxby2001_subj{i}_vt_svm-1/results.json')).iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)],
        'ROI + RFE + CNN': [(acc := pd.read_json(f'result-baseline-rfe/icassp_haxby2001_subj{i}_vt_cnn-1/results.json')).iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)],
        'Yeo + RF + RF': [(acc := pd.read_json(f'result-rf-rf-yeo-RandomSearch/subj{i}/results.json')).iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)],
        'Yeo + RF + SVM': [(acc := pd.read_json(f'result-rf-svm-yeo-RandomSearch/subj{i}/results.json')).iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)],
        'Yeo + RF + CNN': [(acc := pd.read_json(f'result-rf-cnn-yeo-RandomSearch/subj{i}/results.json')).iloc[-1]['Classification report']['accuracy'] for i in range(1, 6)],
    }
    df = pd.DataFrame(data)

    


    # Set the subject column as the index for easier plotting
    df.set_index('Subject', inplace=True)

    # Plotting a bar plot
    ax = df.plot(kind='bar', rot=0)
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Methods on Different Subjects')
    plt.legend(title='Methods', bbox_to_anchor=(1, 1))
    plt.tight_layout()  # Ensures the entire plot is visible
    plt.savefig('bar_plot.png')
    
    plt.clf() 
    # Reshape the data into a long format for box plot
    df_long = df.melt(var_name='Method', value_name='Accuracy')

    # Plotting a box plot
    sns.boxplot(x='Method', y='Accuracy', data=df_long)
    plt.xlabel('Method')
    plt.xticks(rotation=45) # rotate the xticks
    plt.ylabel('Accuracy')
    plt.title('Comparison of Methods on Different Subjects')
    plt.tight_layout()  # Ensures the entire plot is visible
    plt.savefig('box_plot.png')