import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def get_significance_asterisks(p_val):
    """Return asterisks string representing significance level."""
    if p_val < 0.001: return "***"
    elif p_val < 0.01: return "**"
    elif p_val < 0.05: return "*"
    else: return "ns"



if __name__ == '__main__':
    
    # Experiment
    # Experiment = 'Haxby'
    # Experiment = 'Emo'
    # Experiment = 'VKMC'
    # Experiment = 'Language'
    Experiment = 'Gambling'
    
    ################ Subject list ################
    # VKMC
#     Subject_list = ['sub-001',  'sub-003',  'sub-006',  'sub-101',  'sub-103',  'sub-105',  'sub-107',  'sub-202',  'sub-204',  'sub-207',
# 'sub-002',  'sub-005',  'sub-007',  'sub-102',  'sub-104',  'sub-106',  'sub-201',  'sub-203',  'sub-205',  'sub-208',]
    
    # Haxby or Emo
    # Subject_list = ['subj1','subj2','subj3','subj4','subj5']
    
    # Language or Gambling
    Subject_list = ['subj1','subj2','subj3','subj4','subj5','subj6','subj7','subj8','subj9','subj10','subj11','subj12','subj13']
    

    # Example data in a dictionary format
    data = {
        'Subject': Subject_list,
        'ROI-based: RF': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-rf/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list],
        'ROI-based: SVM': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-svm/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list],
        'ROI-based: CNN': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-cnn/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list],
        'ROI-based: ANOVA + RF': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-anova-rf/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        'ROI-based: ANOVA + SVM': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-anova-svm/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        'ROI-based: ANOVA + CNN': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-anova-cnn/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        'ROI-based: RFE + RF': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-rfe-rf/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        'ROI-based: RFE + SVM': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-rfe-svm/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        'ROI-based: RFE + CNN': [(acc := pd.read_csv(f'result-{Experiment}/baseline-roi-rfe-cnn/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        'AutoFMRI: RF + RF': [(acc := pd.read_csv(f'result-{Experiment}/autofmri-rf-rf/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        'AutoFMRI: RF + SVM': [(acc := pd.read_csv(f'result-{Experiment}/autofmri-rf-svm/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
        'AutoFMRI: RF + CNN': [(acc := pd.read_csv(f'result-{Experiment}/autofmri-rf-cnn/{subj}/total_classification_report.csv', index_col=0).loc['accuracy', 'f1-score']) for subj in Subject_list], 
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
    
    
    ############################ add t-test to plot ############################

    # Reshape the data into a long format for box plot
    df_long = df.melt(var_name='Method', value_name='Accuracy')

    # Specify the pairs you want to compare
    selected_pairs = [
        ('AutoFMRI: RF + RF', 'ROI-based: RFE + RF'),
        ('AutoFMRI: RF + SVM', 'ROI-based: RFE + SVM'),
        # ('AutoFMRI: RF + CNN', 'ROI-based: ANOVA + CNN'),
        ('AutoFMRI: RF + CNN', 'ROI-based: CNN'),   
    ]

    # Conduct t-tests for the selected pairs
    p_values = {}
    t_stats = {}
    for pair in selected_pairs:
        t_stat, p_val = stats.ttest_rel(df[pair[0]], df[pair[1]])
        p_values[pair] = p_val
        t_stats[pair] = t_stat

    # Plotting the box plot
    sns.boxplot(x='Method', y='Accuracy', data=df_long)
    plt.xlabel('Method')
    plt.xticks(rotation=45) # rotate the xticks
    plt.ylabel('Accuracy')
    plt.title(f'Comparison of Methods of {Experiment}')

    # Annotate the plot with the t-test results
    y_max = df_long['Accuracy'].max()
    y_min = df_long['Accuracy'].min()
    y_range = y_max - y_min
    offset = y_range * 0.05
    current_y_max = y_max + offset  # Initial vertical position for the first pair's annotation


    methods = df.columns
    for pair, p_val in p_values.items():
        # Get the x positions of the boxes that are being compared
        x1 = methods.get_loc(pair[0])
        x2 = methods.get_loc(pair[1])

        # Add a line between the boxes
        plt.plot([x1, x1, x2, x2], 
                [current_y_max, current_y_max + offset, current_y_max + offset, current_y_max], 
                c='k')

        # Add p-value and significance asterisks
        significance_asterisks = get_significance_asterisks(p_val)
        plt.text((x1 + x2) * 0.5, current_y_max + offset/4, 
                f'{significance_asterisks} (p={p_val:.3f})', 
                ha='center', va='center', fontsize=8)

        # Increase the vertical position for the next pair
        current_y_max += 2*offset

        print(f'{pair[0]} vs {pair[1]}: p={p_val:.3f} t={t_stats[pair]:.3f}')

    plt.ylim(y_min, current_y_max)  # Adjust y limit to make space for all annotations

    plt.tight_layout()
    plt.savefig(f'box_plot_with_ttest_{Experiment}.png')