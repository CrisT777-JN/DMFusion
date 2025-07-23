import pandas as pd

# Load data from an Excel file
for metric in ['SD','MI','VIF','AG','CC','SCD','EN','Qabf','SF']:
    excel_file = '.\Metric\Metric_PET-MRI-low-contrast.xlsx'
    df = pd.read_excel(excel_file, sheet_name=f'{metric}', skiprows=1)

    # Remove rows and columns with NaNs
    df = df.dropna()


    # Compute the column-wise mean
    column_means = df.mean(axis=0)


    print(f"{metric}The mean of each column:")
    with open('output.txt', 'a') as f:
        column_means.to_csv('output.txt',header=False, index=False, sep='\t',mode='a')
        f.write('\n')  
    print(column_means)
