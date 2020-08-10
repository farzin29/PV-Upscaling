from collections import defaultdict
import pandas as pd
import numpy as np
import datetime
import time
import os
import warnings
import re
warnings.filterwarnings('ignore')



postCode = [21, 4800] # adding postcodes
regex_pattern = ''
for enum,_ in enumerate(postCode,start=1):
    if len(str(_))==2:
        regex_pattern += f'(^{_})'
    elif len(str(_))==4:
        regex_pattern += f'({_})'
    if enum != len(postCode):
        regex_pattern += '|'

device_selection_threshold = 0.50
size_threshold = 100000
sample_num,sample_size = 50, 100 # sampling iteration, sample size
input_dir = 'input/'
output_dir = 'output/'
sample_dir = 'sample/'
[os.remove(output_dir+_) for _ in os.listdir(output_dir)];
[os.remove(sample_dir+_) for _ in os.listdir(sample_dir)];
start_time, end_time = datetime.datetime.strptime('07:0','%H:%M').time(), datetime.datetime.strptime('17:30','%H:%M').time()
possible_log_number = (((end_time.hour*60+end_time.minute)-(start_time.hour*60+start_time.minute))/5)+1
seed_number = np.random.randint(1,1000,sample_num)



# part 0
day_system_lst = defaultdict(list)
id_dic = defaultdict(dict)
for each_day in sorted(os.listdir(input_dir),key=lambda x:(int(x.split('-')[0]),int(x.split('-')[1]),int(x.split('-')[2]))):
    col_name = '-'.join(each_day.split('-')[:3])
    df_file = pd.read_csv(input_dir+each_day,header=None)
    df_file.loc[3:,0] = df_file.loc[3:,0].apply(lambda x: datetime.datetime.strptime(x,'%H:%M').time())
    df_file.index = df_file[0]
    df_file.drop(columns=0,inplace=True)
    
    
    df_file.fillna(0,inplace=True)
    df_file = df_file.astype(np.int64)
    series = pd.Series(list(df_file.loc['Postcode',:].astype('str')),index=range(1,len(df_file.columns)+1))
    filtered_series = series.str.extractall(regex_pattern)
    filtered_series_index = [_[0] for _ in filtered_series.index]
    df_file = df_file[filtered_series_index]
    
    for each_col in df_file.columns:
        df_system = df_file.loc[:,[each_col]] # with 2 columns
        if int(df_system.loc['Size'].values[0])>size_threshold:
            continue
        df_system_3_first_rows = df_system.iloc[:3,[0]]
        df_system_part2 = df_system.iloc[3:,[0]]
        df_system_part2_device_period = df_system_part2[(df_system_part2.index >= start_time) & 
                                                       (df_system_part2.index <= end_time) & 
                                                       (df_system_part2[each_col] != 0) &
                                                       (df_system_part2[each_col] != -1)]
        df_system_part2_device_period.dropna(inplace=True)
        day_system_lst[col_name].append(df_system_3_first_rows.loc['SystemID'].values[0])
        id_dic[df_system_3_first_rows.loc['SystemID'].values[0]]['Postcode'] = df_system_3_first_rows.loc['Postcode'].values[0]
        id_dic[df_system_3_first_rows.loc['SystemID'].values[0]]['Size'] = df_system_3_first_rows.loc['Size'].values[0]
        id_dic[df_system_3_first_rows.loc['SystemID'].values[0]][col_name] = len(df_system_part2_device_period)/possible_log_number
id_code_dic = defaultdict(list)
for iD in id_dic.keys():
    for day in day_system_lst.keys():
        if id_code_dic[iD]==[]:
                id_code_dic[iD].append(id_dic[iD]['Postcode'])
                id_code_dic[iD].append(id_dic[iD]['Size'])
        if iD in day_system_lst[day]:
            id_code_dic[iD].append(id_dic[iD][day])
        else:
            id_code_dic[iD].append(0)
df_unique_device = pd.DataFrame(data=list(id_code_dic.values()),index=list(id_code_dic.keys()),
                                columns=['Postcode','Size']+list(day_system_lst.keys()))
df_unique_device['ratio'] = df_unique_device.iloc[:,2:].sum(axis=1)/len(df_unique_device.iloc[:,2:].keys())
df_unique_device.apply(lambda x :round(x,3)).to_excel(output_dir+'ratio.xlsx')




main_df = pd.DataFrame() # store result of each file
main_df_mean = pd.DataFrame()
for each_day in sorted(os.listdir(input_dir),key=lambda x:(int(x.split('-')[0]),int(x.split('-')[1]),int(x.split('-')[2]))):
    print(each_day)
    excel_writer_sample = pd.ExcelWriter(f'{sample_dir}sample_{each_day.split(".")[0]}.xlsx') # save sample item
    time_start = time.time()
    
    
    # part 1
    df_raw = pd.read_csv(input_dir+each_day,header=None)
    df_raw.loc[3:,0] = df_raw.loc[3:,0].apply(lambda x: datetime.datetime.strptime(x,'%H:%M').time())
    df_raw.index = df_raw[0]
    df_raw.drop(columns=0,inplace=True)
    df_raw_3_first_rows = df_raw.iloc[:3]
    df_raw_part2 = df_raw.iloc[3:]
    df_raw_part2_device_period = df_raw_part2[(df_raw_part2.index >= start_time) & 
                                                (df_raw_part2.index <= end_time)]
    df_raw_part2_device_period.replace(0,np.nan,inplace=True)
    df_raw_part2_device_period.replace(-1,np.nan,inplace=True)
    df_raw = pd.concat([df_raw_3_first_rows,df_raw_part2_device_period])
    
    
    df_raw.fillna(0,inplace=True)
    df_raw = df_raw.astype(np.int64)
    # filter columns with postcode = postCode (e.g: postCode = 21)
    series = pd.Series(list(df_raw.loc['Postcode',:].astype('str')),index=range(1,len(df_raw.columns)+1))
    filtered_series = series.str.extractall(regex_pattern)
    filtered_series_index = [_[0] for _ in filtered_series.index]
    df_raw = df_raw[filtered_series_index]
    [df_raw.drop(columns=_,inplace=True) for _ in df_raw.columns if int(df_raw[_]['Size'])>size_threshold];


    # part 2
    df = df_raw.copy() # keep df_raw unchanged for sampling step by creating df and apply changes on it.
    sum_of_recorded_each_row_lst,sum_capacity_lst = [],[]
    for i,row in df.iterrows(): # calculate sum of each row
        if i in ['SystemID','Postcode']:
            continue
        elif i in ['Size']:
            capacity_dic = {col:capacity for col,capacity in zip(df.columns,row)} # getting capacity row
            continue
        else:
            value_dic = {col:value for col,value in zip(df.columns,row)} # getting value of each row
        non_nan = {k:v for k,v in value_dic.items() if np.isnan(v) != True} # drop nan value in each row
        non_nan_capacity = {k:capacity_dic[k] for k in non_nan.keys()} # drop capacity according to nan value in non_nan
        sum_of_recorded_each_row_lst.append(sum(non_nan.values())) # sum of each row except nan values, added to sum_of_recorded_each_row_lst
        sum_capacity_lst.append(sum(non_nan_capacity.values())) # sum of capacity except nan values, added to sum_capacity_lst
    df['sum'] = 3*[np.nan]+sum_of_recorded_each_row_lst # add sum column to df
    df['Size'] = 3*[np.nan]+sum_capacity_lst # add capacity column to df
    df['real_per'] = df['sum']/df['Size'] # add column real_per to df
    
    
    # part 3
    # filter devices that don't satisfy device_selection_threshold
    device_with_device_selection_threshold = list(map(int,df_unique_device[df_unique_device['ratio']>device_selection_threshold].index))
    df_raw_col_with_device_selection_threshold = [_ for _ in device_with_device_selection_threshold if _ in list(map(int,df_raw.loc['SystemID']))]
    df_raw_col_index = [df_raw.loc['SystemID'].index[(df_raw.loc['SystemID'] == _)][0] for _ in df_raw_col_with_device_selection_threshold]
    df_raw = df_raw[df_raw_col_index]

    mae_dic = {} # add mae mean for each sample to mae_dic like {'sample_1':mae_1, 'sample_2':mae_2, ...}
    rmse_dic = {} # add rmse mean for each sample to rmse_dic like {'sample_1':mae_1, 'sample_2':mae_2, ...}
    mae_rmse_df = pd.DataFrame()
    for counter, seed in zip(range(1,sample_num+1),seed_number): # calculate sum of each row for samples like above!
        
        
        df_unique_device_with_threshold = df_unique_device[df_unique_device['ratio']>device_selection_threshold]
        df_raw_SystemID = df_raw.loc['SystemID',:]
        df_unique_device_with_threshold_is_in_df_raw_only = df_unique_device_with_threshold.loc[df_unique_device_with_threshold.index.isin(df_raw_SystemID.values)]
        if len(df_unique_device_with_threshold_is_in_df_raw_only) < sample_size:
            sample_size = len(df_unique_device_with_threshold_is_in_df_raw_only)
        sample = list(df_unique_device_with_threshold_is_in_df_raw_only.sample(n=sample_size,axis=0,replace=False,random_state=seed).index)
        
        sample_col_name = [df_raw.loc['SystemID'].index[(df_raw.loc['SystemID'] == _)][0] for _ in sample]
        df_sample = df_raw[sample_col_name]
        df_sample.to_excel(excel_writer_sample, sheet_name=f'{counter}', header=False)
        sample_sum_lst,sample_capacity_lst = [],[]
        for i,row in df_sample.iterrows():
            if i in ['SystemID','Postcode']:
                continue
            elif i in ['Size']:
                capacity_dic = {_:[] for _ in df_sample.columns}
                {capacity_dic[col].append(capacity) for col,capacity in zip(df_sample.columns,row)}
                continue
            else:
                value_dic = {_:[] for _ in df_sample.columns}
                {value_dic[col].append(value) for col,value in zip(df_sample.columns,row)}
            non_nan = {k:v for k,v in value_dic.items() if np.isnan(v[0]) != True}
            non_nan_capacity = {k:capacity_dic[k] for k in non_nan.keys()}
            sample_sum_lst.append(sum([item for sublist in non_nan.values() for item in sublist]))
            sample_capacity_lst.append(sum([item for sublist in non_nan_capacity.values() for item in sublist]))
        df['sample_'+str(counter)] = 3*[np.nan]+sample_sum_lst # add sample_ column to df
        df['Size_'+str(counter)] = 3*[np.nan]+sample_capacity_lst # add capacity_ column to df
        df['est_per_'+str(counter)] = df['sample_'+str(counter)]/df['Size_'+str(counter)] # add est_per_ column to df
        df.replace('nan%',0,inplace=True) # this is not nan value; it is product of zerodivision.
        df['MAE_'+str(counter)] = abs(df['est_per_'+str(counter)]-df['real_per']) # add MAE_ column to df
        df['RMSE_'+str(counter)] = (df['est_per_'+str(counter)]-df['real_per'])**2 # add RMSE_ column to df
        df['est_per_'+str(counter)] = df['est_per_'+str(counter)].apply(lambda x: f'{x:.2%}') # prettify
        df.loc[['SystemID','Postcode','Size'],['est_per_'+str(counter)]] = 3*[np.nan]
        mae_dic['sample_'+str(counter)] = df['MAE_'+str(counter)].mean()
        rmse_dic['sample_'+str(counter)] = df['RMSE_'+str(counter)].mean()
    excel_writer_sample.save()
    excel_writer_sample.close()
    # # just in case of having something like rand_21.xlsx
    # df['real_per'] = df['real_per'].apply(lambda x: f'{x:.2%}')
    # df.loc[['SystemID','Postcode','Size'],['real_per']] = 3*[np.nan]
    # df.replace('nan%',0,inplace=True)
    # df.to_excel(f'{output_dir}{postCode}_{each_day.split(".")[0]}.xlsx')

    mae_rmse_df.insert(0,'#',list(mae_dic.keys())+['average'])
    mae_rmse_df['MAE'] = list(mae_dic.values())+[np.mean(list(mae_dic.values()))]
    mae_rmse_df['RMSE'] = list(rmse_dic.values())+[np.mean(list(rmse_dic.values()))]
    # mae_rmse_df.to_excel(f'{output_dir}mae_rmse_{each_day.split(".")[0]}.xlsx',index=False)
    mae_rmse_df.index = mae_rmse_df['#']
    mae_rmse_df.drop(columns='#',inplace=True)
    main_df = pd.merge(main_df, mae_rmse_df, left_index=True, right_index=True, how='outer')
    print(f'excecuted time: {time.time()-time_start:.2f} sec')
main_df_mean['MAE'] = main_df.filter(regex='MAE').mean(axis=1)
main_df_mean['RMSE'] = main_df.filter(regex='RMSE').mean(axis=1)**.5
main_df_mean.index = main_df.index
main_df_mean.drop(index='average',inplace=True)
main_df_mean.to_excel(f'{output_dir}output.xlsx')
