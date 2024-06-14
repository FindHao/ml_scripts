import os
import pandas as pd
import openpyxl

# Define the path to the directory containing the CSV files
directory_path = '/home/yhao/p9_clean/logs'

# Define the model libraries and modes
model_libraries = ['torchbench', 'huggingface', 'timm_models']
modes = ['inference', 'training']

# Create a dictionary to hold the data for each sheet
data = {f'{lib}_{mode}': {} for lib in model_libraries for mode in modes}

# Read each CSV file and collect the necessary data
for filename in os.listdir(directory_path):
    if filename.endswith('.csv') and 'accuracy' in filename:
        filepath = os.path.join(directory_path, filename)
        date = filename.split('_')[0][4:]  # Extract date in MMDD format
        df = pd.read_csv(filepath).dropna(how='all')  # Drop empty rows
        
        mode = 'inference' if 'inference' in filename else 'training'
        lib = [lib for lib in model_libraries if lib in filename][0]
        sheet_name = f'{lib}_{mode}'

        if date not in data[sheet_name]:
            data[sheet_name][date] = {}

        for index, row in df.iterrows():
            name = row['name']
            accuracy = row['accuracy']
            if name not in data[sheet_name]:
                data[sheet_name][name] = {}
            data[sheet_name][name][date] = accuracy

# Create a new Excel workbook
excel_path = 'output.xlsx'
writer = pd.ExcelWriter(excel_path, engine='openpyxl')

# Write data to each sheet and add pass rate row
for sheet_name, sheet_data in data.items():
    df = pd.DataFrame(sheet_data).transpose()
    df.index.name = 'name'
    df.reset_index(inplace=True)

    # Calculate pass rates
    pass_rates = {'name': 'Pass Rate'}
    for date in df.columns[1:]:
        pass_count = df[date].apply(lambda x: 1 if 'pass' in str(x).lower() else 0).sum()
        total_count = df[date].count()
        pass_rates[date] = f"{(pass_count / total_count) * 100:.2f}%" if total_count > 0 else '0.00%'

    # Convert pass_rates dictionary to DataFrame
    pass_rate_df = pd.DataFrame([pass_rates])

    # Concatenate the original dataframe with the pass_rate_df
    df = pd.concat([df, pass_rate_df], ignore_index=True)

    # Ensure 'name' column stays first
    columns = ['name'] + [col for col in df.columns if col != 'name']
    df = df[columns]

    # Save to Excel
    df.to_excel(writer, sheet_name=sheet_name, index=False)

# Save the workbook
writer.close()

print(f'Excel file created and saved as {excel_path}')
