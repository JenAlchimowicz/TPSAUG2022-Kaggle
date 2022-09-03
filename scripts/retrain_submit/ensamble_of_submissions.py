import pandas as pd

#######################################
# Read datasets
#######################################

# Mine
sub1 = pd.read_csv('../submissions/submission_7623qk.csv')
sub2 = pd.read_csv('../submissions/submission_14bced.csv')
sub3 = pd.read_csv('../submissions/submission_Ead09.csv')
sub4 = pd.read_csv('../submissions/submission_GG4ak.csv')

# Kaggle
sub5 = pd.read_csv('../submissions/submission_kaggle1.csv')
sub6 = pd.read_csv('../submissions/submission_kaggle2.csv')
sub7 = pd.read_csv('../submissions/submission_kaggle3.csv')

#######################################
# Create ensamble based on quantiles
#######################################

sub = pd.DataFrame(sub1['id'])

sub['failure'] = sub1['failure'].rank(pct=True) * 0.18 + \
                 sub2['failure'].rank(pct=True) * 0.18 + \
                 sub3['failure'].rank(pct=True) * 0.12 + \
                 sub4['failure'].rank(pct=True) * 0.12 + \
                 sub5['failure'].rank(pct=True) * 0.10 + \
                 sub6['failure'].rank(pct=True) * 0.15 + \
                 sub7['failure'].rank(pct=True) * 0.15

sub.to_csv('../submissions/submission_ensamble.csv', index=False)