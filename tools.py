import IPython
import numpy as np
import os 
import pandas as pd
from multiprocessing import Pool 
from IPython.display import display_html
from joblib import Parallel, delayed

### parallel process
def process_parallel(dfs, func):
    res = Parallel(n_jobs=-1)(delayed(func)(df) for df in dfs)
    return pd.concat(res, axis=0)
    
### apply parallel
def apply_parallel(df_grouped, func):
    res = Parallel(n_jobs=-1)(delayed(func)(group) for name, group in df_grouped)
    return pd.concat(res, axis=0)

### multiprocessing
def get_clk_times_list(df):
    pid=os.getpid()
    try:
        temp = tr_log.groupby(['user_id']).apply(aggfunc)
        temp.index = temp.index.droplevel(None)
        temp = temp.reset_index()
    except KeyboardInterrupt:
        print('进程%d被中断...'%pid)    
    return temp

def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    
    print('ppid:%d'%os.getpid())
    try:
        df = pd.concat(pool.map(func, t_split), axis=0)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print('catch keyboardinterupterror')
        pid=os.getpid()
    except Exception as e:
        print(e)
    else:
        print('quit normally')

    return df
###############################

def accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred), "length of y_true and y_pred not equal"
    total_example = len(y_true)
    right_cnt = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            right_cnt += 1
    return right_cnt / total_example

def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    
def on_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

def display(*dfs, head=True, n=5):
    for df in dfs:
        IPython.display.display(df.head(n) if head else df)

## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df

def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df  

def read_data():
    INPUT_DIR = "/kaggle/input" if on_kaggle() else "/home/huawei/risehuang/competition/kaggleWalmart"
    INPUT_DIR = f"{INPUT_DIR}/data"

    print("Reading files...")
    if not os.path.exists(f"{INPUT_DIR}/calendar.pkl"):
        calendar = pd.read_csv(f"{INPUT_DIR}/calendar.csv").pipe(reduce_mem_usage)
        calendar.to_pickle(f"{INPUT_DIR}/calendar.pkl")
    else:
        calendar = pd.read_pickle(f"{INPUT_DIR}/calendar.pkl")
    
    if not os.path.exists(f"{INPUT_DIR}/sell_prices.pkl"):
        sell_prices = pd.read_csv(f"{INPUT_DIR}/sell_prices.csv").pipe(reduce_mem_usage)
        sell_prices.to_pickle(f"{INPUT_DIR}/sell_prices.pkl")
    else:
        sell_prices = pd.read_pickle(f"{INPUT_DIR}/sell_prices.pkl")

    if not os.path.exists(f"{INPUT_DIR}/sales_train_validation.pkl"):
        sales_train_val = pd.read_csv(f"{INPUT_DIR}/sales_train_validation.csv").pipe(
            reduce_mem_usage
        )
        sales_train_val.to_pickle(f"{INPUT_DIR}/sales_train_validation.pkl")
    else:
        sales_train_val = pd.read_pickle(f"{INPUT_DIR}/sales_train_validation.pkl")

    submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

    print("calendar shape:", calendar.shape)
    print("sell_prices shape:", sell_prices.shape)
    print("sales_train_val shape:", sales_train_val.shape)
    print("submission shape:", submission.shape)

    # calendar shape: (1969, 14)
    # sell_prices shape: (6841121, 4)
    # sales_train_val shape: (30490, 1919)
    # submission shape: (60980, 29)

    return calendar, sell_prices, sales_train_val, submission
