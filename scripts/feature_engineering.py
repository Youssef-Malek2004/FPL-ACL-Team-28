import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def one_hot_encode_columns(
    df: pd.DataFrame,
    columns_to_encode: list,
    filename: str = "dataset",
    output_subdir: str = "interim",
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    One-hot encodes specified categorical columns in the given DataFrame.
    Saves both the encoded DataFrame and a record of encoded column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical columns.
    columns_to_encode : list
        List of column names to one-hot encode.
    filename : str, optional
        Base name for saved CSVs (default is 'dataset').
    output_subdir : str, optional
        Folder under /data where outputs will be saved (default is 'interim').
    drop_first : bool, optional
        Whether to drop the first level of each encoded variable
        (useful for regression models to avoid dummy-variable trap).

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame with one-hot encoded columns.
    """

    # --- Setup directories ---
    root_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_folder = os.path.join(root_data_dir, output_subdir)
    os.makedirs(output_folder, exist_ok=True)

    # --- One-hot encode ---
    encoded_df = pd.get_dummies(df, columns=columns_to_encode, drop_first=drop_first)

    # --- Save outputs ---
    encoded_path = os.path.join(output_folder, f"{filename}_encoded.csv")

    encoded_df.to_csv(encoded_path, index=False)

    return encoded_df



def label_encode_column(
    df: pd.DataFrame,
    column: str,
    filename: str = "dataset",
    output_subdir: str = "interim",
) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Label-encodes a single categorical column (e.g., player names).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to label-encode.
    filename : str, optional
        Base name for reference/logging only (default is 'dataset').
    output_subdir : str, optional
        Folder under /data where outputs would be saved if persisted
        (kept here for signature consistency).

    Returns
    -------
    tuple[pd.DataFrame, LabelEncoder]
        - DataFrame with a new column '<column>_encoded'
        - The fitted LabelEncoder (for reverse mapping)
    """

    # Determine absolute path: ../data/interim relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data", output_subdir))

    os.makedirs(data_dir, exist_ok=True)

    le = LabelEncoder()
    df[f"{column}_encoded"] = le.fit_transform(df[column].astype(str))

    df = df.drop(columns=[column])

    print(f"Column '{column}' label-encoded → new column '{column}_encoded'")

    cleaned_path = os.path.join(data_dir, f"{filename}_label_encoded.csv")

    df.to_csv(cleaned_path, index=False)

    return df, le

