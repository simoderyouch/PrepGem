from tqdm import tqdm


def preprocess_cell(fun, pbar):
    pbar.update(1)
    return fun

def preprocess_columns(df, columns, preprocessing_function , fun_name):
    df_copy = df.copy()
    for column in columns:
            print(f"Preprocessing '{column}' with {fun_name}")
            pbar = tqdm(total=df_copy[column].shape[0], desc=f"Preprocessing '{column}' with {fun_name}")
            df_copy[column] = df_copy[column].apply(lambda x: preprocess_cell(preprocessing_function(x), pbar))
            pbar.close()
    return df_copy

