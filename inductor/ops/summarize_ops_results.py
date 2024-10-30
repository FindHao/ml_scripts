import pandas as pd
import os
import glob
from pathlib import Path
import numpy as np

def format_excel(writer, df, sheet_name, is_speedup=True):
    """格式化Excel文件"""
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    
    # 动态计算并设置每列的宽度
    for i, col in enumerate(df.columns):
        # 计算列中最长内容的长度（包括列名和数据）
        column_data = df[col].astype(str)
        max_length = max(
            max(len(str(x)) for x in column_data),  # 数据的最大长度
            len(col)  # 列名的长度
        )
        # 将宽度设置为最大长度的1.2倍以留出一些空间
        worksheet.set_column(i, i, max_length)
    
    # 创建格式
    red_format = workbook.add_format({
        'bg_color': '#FFC7CE',  # 浅红色
        'num_format': '0.00'    # 两位小数
    })
    green_format = workbook.add_format({
        'bg_color': '#C6EFCE',  # 浅绿色
        'num_format': '0.00'    # 两位小数
    })
    normal_format = workbook.add_format({
        'num_format': '0.00'    # 两位小数
    })
    
    # 为所有数值列设置两位小数格式
    for col_num, col_name in enumerate(df.columns):
        if col_name != 'op_name':  # 跳过op_name列
            for row in range(1, len(df) + 1):  # 从1开始以跳过标题行
                worksheet.write(row, col_num, df.iloc[row-1, col_num], normal_format)
    
    # 为比较列添加条件格式
    if is_speedup:
        compare_cols = ['p20_liger_vs_inductor', 'p50_liger_vs_inductor', 'p80_liger_vs_inductor']
    else:
        compare_cols = ['p20_liger_vs_inductor_mem', 'p50_liger_vs_inductor_mem', 'p80_liger_vs_inductor_mem']
    
    for col_name in compare_cols:
        col_idx = df.columns.get_loc(col_name)
        worksheet.conditional_format(1, col_idx, len(df), col_idx, {
            'type': 'cell',
            'criteria': '>=',
            'value': 1.01,
            'format': green_format
        })
        worksheet.conditional_format(1, col_idx, len(df), col_idx, {
            'type': 'cell',
            'criteria': '<=',
            'value': 0.98,
            'format': red_format
        })

def process_folder_results(folder_path):
    """处理单个文件夹的所有CSV文件，返回speedup和memory的汇总数据"""
    print(f"Processing folder: {os.path.basename(folder_path)}")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    speedup_results = []
    memory_results = []
    
    for csv_file in csv_files:
        print(f"Processing CSV file: {csv_file}")
        df = pd.read_csv(csv_file, sep=';')
        
        try:
            # 提取op名称
            op_name = os.path.splitext(os.path.basename(csv_file))[0]
            if op_name.startswith('op_'):
                op_name = op_name[3:]
            
            # 查找相关列
            liger_speedup_col = [col for col in df.columns if 'liger' in col and '-speedup' in col][0]
            inductor_speedup_col = [col for col in df.columns if 'inductor' in col and '-speedup' in col][0]
            liger_mem_col = [col for col in df.columns if 'liger' in col and '-mem_footprint' in col][0]
            inductor_mem_col = [col for col in df.columns if 'inductor' in col and '-mem_footprint' in col][0]
            
            # 转换数据类型
            liger_speedup = pd.to_numeric(df[liger_speedup_col], errors='coerce')
            inductor_speedup = pd.to_numeric(df[inductor_speedup_col], errors='coerce')
            liger_mem = pd.to_numeric(df[liger_mem_col], errors='coerce')
            inductor_mem = pd.to_numeric(df[inductor_mem_col], errors='coerce')
            
            if len(liger_speedup.dropna()) > 0 and len(inductor_speedup.dropna()) > 0:
                # 处理speedup数据
                liger_vs_inductor_speedup = liger_speedup / inductor_speedup
                speedup_metrics = {
                    'op_name': op_name,
                    'p20_liger_speedup': np.percentile(liger_speedup.dropna(), 20),
                    'p50_liger_speedup': np.percentile(liger_speedup.dropna(), 50),
                    'p80_liger_speedup': np.percentile(liger_speedup.dropna(), 80),
                    'p20_inductor_speedup': np.percentile(inductor_speedup.dropna(), 20),
                    'p50_inductor_speedup': np.percentile(inductor_speedup.dropna(), 50),
                    'p80_inductor_speedup': np.percentile(inductor_speedup.dropna(), 80),
                    'p20_liger_vs_inductor': np.percentile(liger_vs_inductor_speedup.dropna(), 20),
                    'p50_liger_vs_inductor': np.percentile(liger_vs_inductor_speedup.dropna(), 50),
                    'p80_liger_vs_inductor': np.percentile(liger_vs_inductor_speedup.dropna(), 80)
                }
                speedup_results.append(speedup_metrics)
            
            if len(liger_mem.dropna()) > 0 and len(inductor_mem.dropna()) > 0:
                # 处理memory数据
                liger_vs_inductor_mem = liger_mem / inductor_mem
                memory_metrics = {
                    'op_name': op_name,
                    'p20_liger_mem': np.percentile(liger_mem.dropna(), 20),
                    'p50_liger_mem': np.percentile(liger_mem.dropna(), 50),
                    'p80_liger_mem': np.percentile(liger_mem.dropna(), 80),
                    'p20_inductor_mem': np.percentile(inductor_mem.dropna(), 20),
                    'p50_inductor_mem': np.percentile(inductor_mem.dropna(), 50),
                    'p80_inductor_mem': np.percentile(inductor_mem.dropna(), 80),
                    'p20_liger_vs_inductor_mem': np.percentile(liger_vs_inductor_mem.dropna(), 20),
                    'p50_liger_vs_inductor_mem': np.percentile(liger_vs_inductor_mem.dropna(), 50),
                    'p80_liger_vs_inductor_mem': np.percentile(liger_vs_inductor_mem.dropna(), 80)
                }
                memory_results.append(memory_metrics)
                
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    return pd.DataFrame(speedup_results), pd.DataFrame(memory_results)

def main():
    base_path = "/tmp/tritonbench"
    
    # 处理每个子文件夹
    for folder in os.listdir(base_path):
        if folder.startswith(('v1', 'results')):
            continue
            
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # 获取该文件夹的所有结果
        speedup_df, memory_df = process_folder_results(folder_path)
        
        # 创建单个Excel文件，包含两个sheet
        output_file = f"{folder}_summary.xlsx"
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            if not speedup_df.empty:
                speedup_df.to_excel(writer, sheet_name='speedup_summary', index=False)
                format_excel(writer, speedup_df, 'speedup_summary', is_speedup=True)
            
            if not memory_df.empty:
                memory_df.to_excel(writer, sheet_name='memory_summary', index=False)
                format_excel(writer, memory_df, 'memory_summary', is_speedup=False)
            
            print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main() 