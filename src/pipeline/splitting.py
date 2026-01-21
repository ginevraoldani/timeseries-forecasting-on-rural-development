import pandas as pd

def split_train_test(df, col_name, train_ratio=0.8):
    """
    Extracts a specific column and splits it into Train and Test sets based on time.
    
    Args:
        df (pd.DataFrame): DataFrame with DatetimeIndex.
        col_name (str): The column name to split.
        train_ratio (float): Proportion of data for training (default 0.8).
        
    Returns:
        tuple: (train_df, test_df). Both are DataFrames with a 'Value' column.
            Returns (None, None) if data is insufficient/empty.
    """
    series = df[col_name].dropna()
    if series.empty:
        print(f"Skipping {col_name}: no data")
        return None, None
    
    n_samples = len(series)
    split_idx = int(n_samples * train_ratio)
    if split_idx == 0 or split_idx >= n_samples:
        print(f"Skipping {col_name}: insufficient data for split (n={n_samples})")
        return None, None
    train = series.iloc[:split_idx].to_frame(name='Value')
    test = series.iloc[split_idx:].to_frame(name='Value')
    return train, test

def split_train_val_test(df):
    """ split time series in train, validation and test set with 70-15-15 ratio
    train set (70%) --> backpropagation
    validation set (15%) --> hyperparameter tuning
    test set (15%) --> performance

    Args:
        df (pd.DataFrame): DataFrame with columns 'Year' and 'Value'

    Returns:
        train: DataFrame containing train set
        val: DataFrame containing validation set
        test: DataFrame containing test set
    """
    df = df.sort_values('Year')
    
    val_start_idx = int(len(df)*0.70)
    test_start_idx = int(len(df)*0.86)

    train = df.iloc[:val_start_idx]                 # 0-70%
    val = df.iloc[val_start_idx:test_start_idx]     # 70-85%
    test = df.iloc[test_start_idx:]                 # 86-100%
    
    print(f"Train: {train['Year'].min()}-{train['Year'].max()} ({len(train)} obs)")
    print(f"Val:   {val['Year'].min()}-{val['Year'].max()} ({len(val)} obs)")
    print(f"Test:  {test['Year'].min()}-{test['Year'].max()} ({len(test)} obs)")
    
    return train, val, test