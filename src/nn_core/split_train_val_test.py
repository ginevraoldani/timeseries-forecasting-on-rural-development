def split_time_series(df):
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