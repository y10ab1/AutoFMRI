import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns




if __name__ == '__main__':
    
    # Experiment
    Experiment = 'Emo'

    ################ Subject list ################
    # VKMC
#     Subject_list = ['sub-001',  'sub-003',  'sub-006',  'sub-101',  'sub-103',  'sub-105',  'sub-107',  'sub-202',  'sub-204',  'sub-207',
# 'sub-002',  'sub-005',  'sub-007',  'sub-102',  'sub-104',  'sub-106',  'sub-201',  'sub-203',  'sub-205',  'sub-208',]
    
    # Haxby or Emo
    Subject_list = ['subj1','subj2','subj3','subj4','subj5']
    

    # Example data in a dictionary format
    data = {
        'Subject': Subject_list,
        'ROI-based: RF': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-rf/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list],
        'ROI-based: SVM': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-svm/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list],
        # 'ROI-based: CNN': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-cnn/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list],
        'ROI-based: ANOVA + RF': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-anova-rf/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        'ROI-based: ANOVA + SVM': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-anova-svm/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        # 'ROI-based: ANOVA + CNN': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-anova-cnn/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        'ROI-based: RFE + RF': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-rfe-rf/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        'ROI-based: RFE + SVM': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-rfe-svm/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        # 'ROI-based: RFE + CNN': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-rfe-cnn/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        'AutoFMRI: RF + RF': [(acc := pd.read_csv(f'result-{Experiment}/autofmri-rf-rf/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        'AutoFMRI: RF + SVM': [(acc := pd.read_csv(f'result-{Experiment}/autofmri-rf-svm/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        # 'AutoFMRI: RF + CNN': [(acc := pd.read_csv(f'result-{Experiment}/autofmri-rf-cnn/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
    }
    df = pd.DataFrame(data)

    
    

    # Set the subject column as the index for easier plotting
    df.set_index('Subject', inplace=True)

    # Plotting a bar plot
    ax = df.plot(kind='bar', rot=0)
    plt.xlabel('Subject')
    plt.xticks(rotation=45) # rotate the xticks
    plt.ylabel('Accuracy')
    plt.title(f'Comparison of Methods of {Experiment}')
    plt.legend(title='Methods', bbox_to_anchor=(1, 1))
    plt.tight_layout()  # Ensures the entire plot is visible
    plt.savefig(f'bar_plot_{Experiment}.png')
    
    plt.clf() 
    # Reshape the data into a long format for box plot
    df_long = df.melt(var_name='Method', value_name='Accuracy')

    # Plotting a box plot
    sns.boxplot(x='Method', y='Accuracy', data=df_long)
    plt.xlabel('Method')
    plt.xticks(rotation=45) # rotate the xticks
    plt.ylabel('Accuracy')
    plt.title(f'Comparison of Methods of {Experiment}')
    plt.tight_layout() # Ensures the entire plot is visible
    plt.savefig(f'box_plot_{Experiment}.png')