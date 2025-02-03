# %%
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, Normalizer, MinMaxScaler

# %%
def find_correlated_pairs_amount(df: pd.DataFrame, threshold=0.9)->int:
    """
    Finds the number of highly correlated column pairs in a DataFrame.
    
    Parameters:
    - df: pd.DataFrame
        The input DataFrame to analyze.
    - threshold: float
        The correlation threshold above which columns are considered highly correlated.
        
    Returns:
    - int
        The number of column pairs with a correlation greater than the threshold, this pairs may be a huge number because it double count many 
        individual ones. 
    """
    # Compute the correlation matrix
    corr_matrix = df.corr()
    
    # Mask the upper triangle of the correlation matrix
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_triangle = corr_matrix.where(mask)
    
    # Find column pairs with correlation above the threshold
    correlated_pairs = [
        (col1, col2) 
        for col1 in upper_triangle.columns 
        for col2 in upper_triangle.index 
        if abs(upper_triangle.loc[col2, col1]) > threshold
    ]
    
    return len(correlated_pairs)

# %%
def find_unique_correlated_groups(df: pd.DataFrame, threshold=0.9):
    """
    Finds the unique groups of highly correlated columns in a DataFrame.
    
    Parameters:
    - df: pd.DataFrame
        The input DataFrame to analyze.
    - threshold: float
        The correlation threshold above which columns are considered highly correlated.
        
    Returns:
    - int
        The number of unique groups of correlated columns.
    - list
        A list of unique correlated groups.
    """
    # Compute the correlation matrix
    corr_matrix = df.corr()
    
    # Identify pairs of columns with correlation above the threshold
    correlated_pairs = [
        (col1, col2) 
        for col1 in corr_matrix.columns 
        for col2 in corr_matrix.columns 
        if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > threshold
    ]
    
    # Build a graph where nodes are columns and edges indicate high correlation
    graph = nx.Graph()
    graph.add_edges_from(correlated_pairs)
    
    # Find connected components in the graph (unique groups of correlated columns)
    correlated_groups = list(nx.connected_components(graph))
    
    return len(correlated_groups), correlated_groups

# %%
def find_null_all_data(df: pd.DataFrame)->int:
    '''
    find the number of nulls across all dataset
    - df: the dataframe
    '''
    return df.isnull().sum()

# %%
def find_null_cols(df: pd.DataFrame, str_lst: list[str]):
    '''
    find the number of nulls of some columns
    - df: the dataframe
    - str_lst: input the column names in list of string
    '''
    valid_columns = [col for col in str_lst if col in df.columns]

    # Warn if some columns are invalid
    invalid_columns = [col for col in str_lst if col not in df.columns]
    if invalid_columns:
        print(f"Warning: The following columns are not in the DataFrame: {invalid_columns}")
    return df[valid_columns].isnull().sum()

# %%
def seperate_num_and_non_num(df: pd.DataFrame)->pd.DataFrame:
    """
    Separate the dataset into two DataFrames:
    one with numeric columns (others filled with NaN)
    and one with non-numeric columns (others filled with NaN).

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    tuple: A tuple containing two DataFrames - numeric and non-numeric, preserving column names.
    """
    df_numeric = df.apply(lambda col: col if pd.api.types.is_numeric_dtype(col) else pd.NA)
    df_non_numeric = df.apply(lambda col: col if not pd.api.types.is_numeric_dtype(col) else pd.NA)
    return df_numeric, df_non_numeric

# %%
def drop_correlated_columns(df: pd.DataFrame, threshold, exclude_word: list[str] = None) -> pd.DataFrame:
    """
    Drops columns from groups of highly correlated columns until only one remains per group,
    ignoring columns that contain specified words in their names.

    Parameters:
    - df: pd.DataFrame
        The input DataFrame to analyze and modify.
    - threshold: float
        The correlation threshold above which columns are considered highly correlated.
    - exclude_word: list[str]
        A list of words to exclude from consideration for dropping. Columns containing these
        words (case-insensitively) are not dropped.

    Returns:
    - pd.DataFrame
        A modified DataFrame with reduced columns.
    """
    exclude_columns = set()
    if exclude_word:
        exclude_columns = {
            col for col in df.columns
            if any(word.lower() in col.lower() for word in exclude_word)
        }
    corr_matrix = df.corr()
    correlated_pairs = [
        (col1, col2) 
        for col1 in corr_matrix.columns 
        for col2 in corr_matrix.columns 
        if col1 != col2 
        and abs(corr_matrix.loc[col1, col2]) > threshold
        and col1 not in exclude_columns
        and col2 not in exclude_columns
    ]

    graph = nx.Graph()
    graph.add_edges_from(correlated_pairs)
    correlated_groups = list(nx.connected_components(graph))
    
    columns_to_keep = {list(group)[0] for group in correlated_groups}
    uncorrelated_columns = set(df.columns) - set(graph.nodes)
    columns_to_keep.update(uncorrelated_columns)
    columns_to_keep.update(exclude_columns) 
    
    return df[list(columns_to_keep)]


# %%
def exclude_columns(df: pd.DataFrame, word_lst:list[str])->pd.DataFrame:
    '''
    Exclude all columns in the dataset if they contain specific words
    - df: the dataframe
    - word_lst: list of words to exclude if they are found in any column names
    '''
    df_new:pd.DataFrame = df.loc[:, ~df.columns.str.contains('|'.join(word_lst), case = False)]
    return df_new

def include_columns(df:pd.DataFrame, word_lst:list[str]) -> pd.DataFrame:
    '''
        Include all columns in the dataset if they contain specific words
        - df: the dataframe
        - word_lst: list of words to include if they are found in any column names
    '''
    df_new: pd.DataFrame = df.loc[:, df.columns.str.contains('|'.join(word_lst), case=False)]
    return df_new

# %%
def KNN_single_columns_imputation_numeric(df: pd.DataFrame, exclude_columns: list[str] = None, n_neighbors: int = 5)->pd.DataFrame:
    """
    Runs KNN on numeric columns in the DataFrame to handle missing values or other tasks, skipping non-numeric data.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    exclude_columns (list[str], optional): A list of column names to exclude from KNN. Default is None.
    n_neighbors (int): Number of neighbors for KNN. Default is 5.

    Returns:
    pd.DataFrame: DataFrame with numeric columns updated using KNN.
    """
    exclude_columns = exclude_columns or []
    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_columns]
    if not numeric_columns:
        return df
    imputer  = KNNImputer(n_neighbors=n_neighbors)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df


# %%


def standarize_dataframe(df: pd.DataFrame, exclude_words: list[str] = None, method: str = 'z-score', norm: str = None) -> pd.DataFrame:
    '''
    Standardizes or normalizes numeric columns in a DataFrame, with additional support for max-abs scaling and log transformation.

    This function provides multiple methods for scaling or normalizing numeric data:

    - **z-score**: Standardizes data by centering it around the mean (0) and scaling it to unit variance (standard deviation of 1). Best for data assumed to be normally distributed or needed for algorithms sensitive to variances (e.g., PCA, linear regression).
    - **min-max**: Scales data to a specified range, usually [0, 1]. Ideal for feature scaling in distance-based algorithms like k-NN or neural networks where bounded ranges are helpful.
    - **robust**: Scales data using the median and interquartile range, reducing the influence of outliers. Useful when data contains significant outliers.
    - **max-abs**: Scales data by dividing each value by the maximum absolute value. Suitable for sparse data to preserve zero entries.
    - **log**: Applies a logarithmic transformation (using log1p to handle zero values). Effective for compressing large value ranges and handling skewed data.

    Additionally, normalization (L1 or L2) is available for row-wise scaling:

    - **L1**: Scales rows so the sum of absolute values equals 1. Useful for sparse data or when focusing on proportions.
    - **L2**: Scales rows so the Euclidean norm (square root of sum of squared values) equals 1. Often used in machine learning models sensitive to vector magnitude, like SVMs.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    exclude_words (list[str]): List of words to exclude columns containing them (case-insensitive). Default is None.
    method (str): Standardization method ('z-score', 'min-max', 'robust', 'max-abs', 'log'). Default is 'z-score'.
    norm (str): Normalization method ('l1', 'l2'). Default is None.

    Returns:
    pd.DataFrame: DataFrame with standardized or normalized columns.

    Raises:
    ValueError: If an unsupported method or normalization type is provided.
    '''
    # Identify columns to exclude based on words
    exclude_columns = set()
    if exclude_words:
        exclude_columns = {
            col for col in df.columns
            if any(word.lower() in col.lower() for word in exclude_words)
        }

    # Columns to standardize
    cols_to_standarize = [
        col for col in df.select_dtypes(include='number').columns if col not in exclude_columns
    ]

    if not cols_to_standarize:
        return df

    if norm:
        normalizer = Normalizer(norm=norm)
        df[cols_to_standarize] = normalizer.fit_transform(df[cols_to_standarize])
        return df

    if method == 'z-score':
        scaler = StandardScaler()
    elif method == 'min-max':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'max-abs':
        scaler = MaxAbsScaler()
    elif method == 'log':
        df[cols_to_standarize] = df[cols_to_standarize].apply(lambda x: np.log1p(x))
        return df
    else:
        raise ValueError(f"Unsupported method '{method}'. Choose 'z-score', 'min-max', 'robust', 'max-abs', or 'log'.")
    
    df[cols_to_standarize] = scaler.fit_transform(df[cols_to_standarize])
    return df


def drop_null_columns(df: pd.DataFrame, threshold: float = 0.8, word_lst: list[str] = None) -> pd.DataFrame:
    """
    Drops columns with a proportion of null values exceeding a given threshold, 
    except for columns containing words from a specified list (case-insensitive).

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    threshold (float): The proportion of null values above which a column is dropped. Default is 0.8.
    word_lst (list[str]): List of words to exclude columns containing them (case-insensitive). Default is None.

    Returns:
    pd.DataFrame: A modified DataFrame with dropped columns based on the null value threshold.
    """
    if word_lst is None:
        word_lst = []

    exclude_columns = {
        col for col in df.columns if any(word.lower() in str(col).lower() for word in word_lst)
    }
    cols_to_drop = []
    for col in df.columns:
        if col not in exclude_columns:
            try:
                null_proportion = df[col].isnull().mean()
                if null_proportion > threshold:
                    cols_to_drop.append(col)
            except Exception as e:
                print(f"Skipping column '{col}' due to error: {e}")

    return df.drop(columns=cols_to_drop, errors='ignore')


def show_all_unique_inputs(df: pd.DataFrame, column_name: str) -> list:
    """
    Displays all unique inputs in a specified column as a list.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column name to extract unique values from.

    Returns:
        list: A list of unique values from the specified column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    return df[column_name].dropna().unique().tolist()

def force_nan(df: pd.DataFrame, value:any) -> None:
    df.replace(value, np.nan, inplace=True)


# %%

