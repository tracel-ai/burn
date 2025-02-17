import argparse

import pyarrow as pa
from datasets import Audio, Image, load_dataset
from sqlalchemy import Column, Integer, Table, create_engine, event, inspect
from sqlalchemy.types import LargeBinary


def download_and_export(
    name: str,
    subset: str,
    db_file: str,
    token: str,
    cache_dir: str,
    data_dir: str | None,
    trust_remote_code: bool,
):
    """
    Download a dataset from using HuggingFace dataset and export it to a sqlite database.
    """

    # TODO For media columns (Image and Audio) sometimes when decode=False,
    # bytes can be none {'bytes': None, 'path': 'healthy_train.265.jpg'}
    # We should handle this case, but unfortunately we did not come across this case yet to test it.

    print("*" * 80)
    print("Starting huggingface dataset download and export")
    print(f"Dataset Name: {name}")
    print(f"Subset Name: {subset}")
    print(f"Sqlite database file: {db_file}")
    print(f"Trust remote code: {trust_remote_code}")
    if cache_dir is None:
        print(f"Custom cache dir: {cache_dir}")
    print("*" * 80)

    # Load the dataset
    dataset_all = load_dataset(
        name,
        subset,
        cache_dir=cache_dir,
        data_dir=data_dir,
        use_auth_token=token,
        trust_remote_code=trust_remote_code,
    )

    print(f"Dataset: {dataset_all}")

    # Create the database connection descriptor (sqlite)
    engine = create_engine(f"sqlite:///{db_file}")

    # Set some sqlite pragmas to speed up the database
    event.listen(engine, "connect", set_sqlite_pragma)

    # Add an row_id column to each table as primary key (datasets does not have API for this)
    event.listen(Table, "before_create", add_pk_column)

    # Export each split in the dataset
    for key in dataset_all.keys():
        dataset = dataset_all[key]

        # Disable decoding for audio and image fields
        dataset = disable_decoding(dataset)

        # Flatten the dataset
        dataset = dataset.flatten()

        # Rename columns to remove dots from the names
        dataset = rename_columns(dataset)

        print(f"Saving dataset: {name} - {key}")
        print(f"Dataset features: {dataset.features}")

        # Save the dataset to a sqlite database
        dataset.to_sql(
            key,  # table name
            engine,
            # don't save the index, use row_id instead (index is not unique)
            index=False,
            dtype=blob_columns(dataset),  # save binary columns as blob
        )

    # Print the schema of the database so we can reference the columns in the rust code
    print_table_info(engine)


def disable_decoding(dataset):
    """
    Disable decoding for audio and image fields. The fields will be saved as raw file bytes.
    """
    for k, v in dataset.features.items():
        if isinstance(v, Audio):
            dataset = dataset.cast_column(k, Audio(decode=False))
        elif isinstance(v, Image):
            dataset = dataset.cast_column(k, Image(decode=False))

    return dataset


def rename_columns(dataset):
    """
    Rename columns to remove dots from the names. Dots appear in the column names because of the flattening.
    Dots are not allowed in column names in rust and sql (unless quoted). So we replace them with underscores.
    This way there is an easy name mapping between the rust and sql columns.
    """

    for name in dataset.features.keys():
        if "." in name:
            dataset = dataset.rename_column(name, name.replace(".", "_"))

    return dataset


def blob_columns(dataset):
    """
    Make sure all binary columns are blob columns in the database because
    `to_sql` exports binary values as TEXT instead of BLOB.
    """
    type_mapping = {}
    for name, value in dataset.features.items():
        if value.pa_type is not None and pa.types.is_binary(value.pa_type):
            type_mapping[name] = LargeBinary
    return type_mapping


def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Set some sqlite pragmas to speed up the database
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = OFF")
    cursor.close()


def add_pk_column(target, connection, **kw):
    """
    Add an id column to each table.
    """
    target.append_column(Column("row_id", Integer, primary_key=True))


def print_table_info(engine):
    """
    Print the schema of the database so we can reference the columns in the rust code
    """
    print(f"Printing table schema for sqlite3 db ({engine})")
    inspector = inspect(engine)
    for table_name in inspector.get_table_names():
        print(f"Table: {table_name}")
        for column in inspector.get_columns(table_name):
            print(f"Column: {column['name']} - {column['type']}")
        print("")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Huggingface datasets downloader to use with burn-dataset"
    )
    parser.add_argument(
        "--name", type=str, help="Name of the dataset to download", required=True
    )
    parser.add_argument(
        "--file", type=str, help="Base file name where the data is saved", required=True
    )
    parser.add_argument(
        "--subset", type=str, help="Subset name", required=False, default=None
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace authentication token",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--cache_dir", type=str, help="Cache directory", required=False, default=None
    )
    parser.add_argument(
            "--data_dir", type=str, help="Relative path to a specific subset of your dataset", required=False, default=None
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        help="Trust remote code",
        required=False,
        default=None,
    )

    return parser.parse_args()


def run():
    args = parse_args()

    download_and_export(
        args.name,
        args.subset,
        args.file,
        args.token,
        args.data_dir,
        args.cache_dir,
        args.trust_remote_code,
    )


if __name__ == "__main__":
    run()
