use dirs::home_dir;
use std::fs;
use std::marker::PhantomData;
use std::{fs::create_dir_all, path::Path};

use crate::Dataset;

use r2d2::Pool;
use r2d2_sqlite::rusqlite::OptionalExtension;
use r2d2_sqlite::{rusqlite::OpenFlags, SqliteConnectionManager};
use sanitize_filename::sanitize;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_rusqlite::*;

/// Dataset where all items are stored in a sqlite database.
///
/// Note: The database must have a table with the same name as the split.
/// The table must have a primary key column named `row_id` which is used to index the rows.
/// `row_id` starts with 1 (one) and `index` starts with 0 (zero) (`row_id` = `index` + 1).
/// The column names must match the field names of the <I> struct. However, the field names
/// can be a subset of column names and can be in any order.
///
/// Supported serialization field types: https://docs.rs/serde_rusqlite/latest/serde_rusqlite and
/// Sqlite3 types: https://www.sqlite.org/datatype3.html
///
/// Item struct example:
///
/// ```rust
///
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// pub struct Sample {
///    column_str: String,     // column name: column_str with type TEXT
///    column_bytes: Vec<u8>,  // column name: column_bytes with type BLOB
///    column_int: i64,        // column name: column_int with type INTEGER
///    column_bool: bool,      // column name: column_bool with type INTEGER
///    column_float: f64,      // column name: column_float with type REAL
/// }
/// ```
///
/// Sqlite table example:
///
/// ```sql
///
/// CREATE TABLE train (
///     column_str TEXT,
///     column_bytes BLOB,
///     column_int INTEGER,
///     column_bool BOOLEAN,
///     column_float FLOAT,
///     row_id INTEGER NOT NULL,
///     PRIMARY KEY (row_id)
/// );
///
/// ```
#[derive(Debug)]
pub struct SqliteDataset<I> {
    db_file: String,
    split: String,
    conn_pool: Pool<SqliteConnectionManager>,
    columns: Vec<String>,
    len: usize,
    select_statement: String,
    row_serialized: bool,
    phantom: PhantomData<I>,
}

impl<I> SqliteDataset<I> {
    /// Initializes a `SqliteDataset` from a SQLite database file.
    ///
    /// This function sets up a `SqliteDataset` to read from a specific table in the SQLite database file.
    /// The chosen table must share the same name as `split`, and must contain a primary key column named `row_id`.
    ///
    /// # Arguments
    ///
    /// * `db_file`: &str - The path to the SQLite database file.
    ///
    /// * `split`: &str - The name of the table in the database that the dataset is supposed to represent.
    ///
    /// * `row_serialized`: bool - This argument indicates the structure of the table. If `true`, each row in the table is expected
    /// to be serialized using the MessagePack format in a single `item` column. If `false`, each field of the item
    /// is expected to be stored in its own column. Serialization of the entire item, including all fields, is necessary because
    /// there are complex fields that cannot be mapped to a SQLite type, such as nested structs, vectors, etc. However,
    /// if the item is a simple struct, e.g., a struct with only primitive fields, then it is possible to store each
    /// field in its own column, and set `row_serialized` to `false`. This type of SQLite table is easier to work with and
    /// analyze using external tools, such as DB Browser for SQLite.
    ///
    /// # Returns
    ///
    /// Returns a new instance of `SqliteDataset`, configured to read from the specified SQLite database file and table.
    ///
    /// # Panics
    ///
    /// This function might panic if the database file does not exist, or if the specified table does not exist in the database,
    /// or does not have the expected structure.
    pub fn from_db_file(db_file: &str, split: &str, row_serialized: bool) -> Self {
        // Create a connection pool
        let conn_pool = create_conn_pool(db_file, false);

        // Create a select statement and save it
        let select_statement = if row_serialized {
            format!("select item from {split} where row_id = ?")
        } else {
            format!("select * from {split} where row_id = ?")
        };

        // Save the column names and the number of rows
        let (columns, len) = fetch_columns_and_len(&conn_pool, &select_statement, split);

        SqliteDataset {
            db_file: db_file.to_string(),
            split: split.to_string(),
            conn_pool,
            columns,
            len,
            select_statement,
            row_serialized,
            phantom: PhantomData::default(),
        }
    }

    /// Get the database file name.
    pub fn db_file(&self) -> &str {
        self.db_file.as_str()
    }

    /// Get the split name.
    pub fn split(&self) -> &str {
        self.split.as_str()
    }
}

impl<I> Dataset<I> for SqliteDataset<I>
where
    I: Clone + Send + Sync + DeserializeOwned,
{
    /// Get an item from the dataset.
    fn get(&self, index: usize) -> Option<I> {
        // Row ids start with 1 (one) and index starts with 0 (zero)
        let row_id = index + 1;

        // Get a connection from the pool
        let connection = self.conn_pool.get().unwrap();
        let mut statement = connection.prepare(self.select_statement.as_str()).unwrap();

        if self.row_serialized {
            // Fetch with a single column `item` and deserialize it with MessagePack
            statement
                .query_row([row_id], |row| {
                    // Deserialize item (blob) with MessagePack (rmp-serde)
                    Ok(
                        rmp_serde::from_slice::<I>(row.get_ref(0).unwrap().as_blob().unwrap())
                            .unwrap(),
                    )
                })
                .optional() //Converts Error (not found) to None
                .unwrap()
        } else {
            // Fetch a row with multiple columns and deserialize it serde_rusqlite
            statement
                .query_row([row_id], |row| {
                    // Deserialize the row with serde_rusqlite
                    Ok(from_row_with_columns::<I>(row, &self.columns).unwrap())
                })
                .optional() //Converts Error (not found) to None
                .unwrap()
        }
    }

    /// Return the number of rows in the dataset.
    fn len(&self) -> usize {
        self.len
    }
}

/// Fetch the column names and the number of rows from the database.
fn fetch_columns_and_len(
    conn_pool: &Pool<SqliteConnectionManager>,
    select_statement: &str,
    split: &str,
) -> (Vec<String>, usize) {
    // Save the column names
    let connection = conn_pool.get().unwrap();
    let statement = connection.prepare(select_statement).unwrap();
    let columns = columns_from_statement(&statement);

    // Count the number of rows and save it as len
    let mut statement = connection
        .prepare(format!("select count(*) from {split}").as_str())
        .unwrap();
    let len = statement
        .query_row([], |row| {
            let len: usize = row.get(0)?;
            Ok(len)
        })
        .unwrap();
    (columns, len)
}

/// Create a connection pool and make sure the connections are read only
fn create_conn_pool(db_file: &str, create_db: bool) -> Pool<SqliteConnectionManager> {
    let sqlite_flags = if create_db {
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE
    } else {
        OpenFlags::SQLITE_OPEN_READ_ONLY
    };

    // Create a connection pool and make sure the connections are read only
    let manager = SqliteConnectionManager::file(db_file).with_flags(sqlite_flags);
    let conn_pool: Pool<SqliteConnectionManager> = Pool::new(manager).unwrap();
    conn_pool
}

pub struct SqliteDatasetSaver<I> {
    name: String,
    db_file: Option<String>,
    splits: Option<Vec<String>>,
    overwrite: bool,
    base_dir: Option<String>,
    conn_pool: Option<Pool<SqliteConnectionManager>>,
    is_initialized: bool,
    phantom: PhantomData<I>,
}

impl<I> SqliteDatasetSaver<I>
where
    I: Clone + Send + Sync + Serialize + DeserializeOwned,
{
    /// Create a new SqliteDatasetSaver.
    pub fn new(name: &str) -> Self {
        SqliteDatasetSaver {
            name: name.to_string(),
            db_file: None,
            splits: None,
            base_dir: None,
            overwrite: false,
            conn_pool: None,
            is_initialized: false,
            phantom: PhantomData::default(),
        }
    }

    /// Set the splits.
    pub fn with_splits(mut self, splits: &[&str]) -> Self {
        self.splits = Some(splits.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Check if the dataset exists.
    pub fn exists(&self) -> bool {
        let db_file = self.db_file();
        Path::new(&db_file).exists()
    }

    /// Return the SqliteDataset for the given split.
    pub fn dataset(&self, split: &str) -> SqliteDataset<I> {
        if !self.exists() {
            panic!("Dataset does not exist: {}", self.name);
        }

        SqliteDataset::from_db_file(self.db_file().as_str(), split, true)
    }

    /// Initialize the dataset saver by creating the database file, tables and connection pool.
    pub fn init(mut self) -> Self {
        // Make sure the splits are set
        if self.splits.is_none() {
            panic!("Splits not set");
        }

        // Create the database file
        let db_file = self.db_file();
        if Path::new(&db_file).exists() {
            if self.overwrite {
                fs::remove_file(&db_file).unwrap();
            } else {
                panic!("Database file already exists: {}", db_file);
            }
        }
        self.db_file = Some(db_file.to_string());

        // Create a connection pool
        let conn_pool = create_conn_pool(&db_file, true);
        self.conn_pool = Some(conn_pool);

        // Create tables
        self.create_tables();

        // Set is_initialized to true
        self.is_initialized = true;

        self
    }

    /// Serialize and save an item to the database.
    ///
    /// # Returns
    /// The index of the inserted row.
    pub fn save(&self, split: &str, item: &I) -> usize {
        // Make sure the dataset saver is initialized
        if !self.is_initialized {
            panic!("Dataset saver not initialized");
        }

        if !self.splits.as_ref().unwrap().contains(&split.to_string()) {
            panic!("Split not found: {}", split);
        }

        // Get a connection from the pool
        let conn_pool = self.conn_pool.as_ref().unwrap();
        let connection = conn_pool.get().unwrap();
        let insert_statement = format!("insert into {split} (item) values (?)", split = split);

        // Serialize the item using MessagePack
        let serialized_item = rmp_serde::to_vec(item).unwrap();

        // Insert the serialized item into the database
        connection
            .execute(insert_statement.as_str(), [serialized_item])
            .unwrap();

        // Get the primary key of the last inserted row and convert to index (row_id-1)
        (connection.last_insert_rowid() - 1) as usize
    }

    /// Create tables for each split.
    fn create_tables(&self) {
        let conn_pool = self.conn_pool.as_ref().unwrap();

        for split in &self.splits.as_ref().unwrap().to_vec() {
            let connection = conn_pool.get().unwrap();
            let create_table_statement = format!(
                "create table {split} (row_id integer primary key autoincrement not null, item blob not null)"
            );
            connection
                .execute(create_table_statement.as_str(), [])
                .unwrap();
        }
    }

    /// Set the base directory to store the dataset (default: $HOME/.cache/burn-dataset).
    pub fn with_base_dir(mut self, base_dir: &str) -> Self {
        self.base_dir = Some(base_dir.to_string());
        self
    }

    /// Overwrite the database file if it already exists.
    pub fn with_overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite = overwrite;
        self
    }

    /// Get the database file path.
    pub fn db_file(&self) -> String {
        let base_dir = base_dir(self.base_dir.clone());
        let name = sanitize(self.name.clone());
        format!("{base_dir}/{name}.db")
    }
}

/// Determine the base directory to store the dataset.
pub fn base_dir(base_dir: Option<String>) -> String {
    let base_dir = if let Some(base_dir) = base_dir {
        base_dir
    } else {
        let home_dir = home_dir().unwrap();
        let home_dir = home_dir.to_str().unwrap();
        let cache_dir = format!("{home_dir}/.cache/burn-dataset");
        cache_dir
    };

    create_dir_all(&base_dir).ok();

    base_dir
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rayon::prelude::*;
    use rstest::{fixture, rstest};
    use serde::{Deserialize, Serialize};
    use tempfile::{tempdir, TempDir};

    use super::*;

    type SqlDs = SqliteDataset<Sample>;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct Sample {
        column_str: String,
        column_bytes: Vec<u8>,
        column_int: i64,
        column_bool: bool,
        column_float: f64,
    }

    #[fixture]
    fn train_dataset() -> SqlDs {
        SqliteDataset::from_db_file("tests/data/sqlite-dataset.db", "train", false)
    }

    #[rstest]
    pub fn len(train_dataset: SqlDs) {
        assert_eq!(train_dataset.len(), 2);
    }

    #[rstest]
    pub fn get_some(train_dataset: SqlDs) {
        let item = train_dataset.get(0).unwrap();
        assert_eq!(item.column_str, "HI1");
        assert_eq!(item.column_bytes, vec![55, 231, 159]);
        assert_eq!(item.column_int, 1);
        assert!(item.column_bool);
        assert_eq!(item.column_float, 1.0);
    }

    #[rstest]
    pub fn get_none(train_dataset: SqlDs) {
        assert_eq!(train_dataset.get(10), None);
    }

    #[rstest]
    pub fn multi_thread(train_dataset: SqlDs) {
        let indices: Vec<usize> = vec![0, 1, 1, 3, 4, 5, 6, 0, 8, 1];
        let results: Vec<Option<Sample>> =
            indices.par_iter().map(|&i| train_dataset.get(i)).collect();

        let mut match_count = 0;
        for (_index, result) in indices.iter().zip(results.iter()) {
            match result {
                Some(_val) => match_count += 1,
                None => (),
            }
        }

        assert_eq!(match_count, 5);
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct Complex {
        column_str: String,
        column_bytes: Vec<u8>,
        column_int: i64,
        column_bool: bool,
        column_float: f64,
        colomn_complex: Vec<Vec<Vec<[u8; 3]>>>,
    }

    /// Create a temporary directory.
    #[fixture]
    fn tmp_dir() -> TempDir {
        // Create a TempDir. This object will be automatically
        // deleted when it goes out of scope.
        tempdir().unwrap()
    }
    type SqlDsSaver = SqliteDatasetSaver<Complex>;

    /// Create a SqliteDatasetSaver with a temporary directory.
    /// Make sure to return the temporary directory so that it is not deleted.
    #[fixture]
    fn dataset_fixture(tmp_dir: TempDir) -> (SqlDsSaver, TempDir) {
        let temp_dir_str = tmp_dir.path().to_str().unwrap().to_string();
        (
            SqliteDatasetSaver::<Complex>::new("preprocessed")
                .with_base_dir(temp_dir_str.as_str())
                .with_splits(&["train", "test"])
                .init(),
            tmp_dir,
        )
    }

    #[test]
    pub fn sqlite_dataset_saver_new() {
        let random_name = format!("dataset-{}", rand::random::<u32>());
        let ds_saver = SqliteDatasetSaver::<Complex>::new(random_name.as_str());

        assert_eq!(ds_saver.name, random_name.as_str());
        assert_eq!(ds_saver.base_dir, None);
        assert!(!ds_saver.overwrite);
        assert!(!ds_saver.is_initialized);
        assert_eq!(ds_saver.splits, None);
        assert!(ds_saver.conn_pool.is_none());
        assert!(ds_saver
            .db_file()
            .ends_with(format!("{}.db", random_name).as_str()));
        assert!(
            !ds_saver.exists(),
            "db file {} should not exist (remove it manually if it does)",
            ds_saver.db_file()
        );
    }

    #[rstest]
    pub fn sqlite_dataset_saver_init(tmp_dir: TempDir) {
        let temp_dir_str = tmp_dir.path().to_str().unwrap().to_string();

        let ds_saver = SqlDsSaver::new("preprocessed")
            .with_base_dir(temp_dir_str.as_str())
            .with_splits(&["train", "test"])
            .init();

        assert_eq!(ds_saver.base_dir, Some(temp_dir_str));

        assert!(ds_saver.is_initialized);
        assert_eq!(
            ds_saver.splits,
            Some(vec!["train".to_string(), "test".to_string()])
        );
        assert!(ds_saver.conn_pool.is_some());
        assert!(ds_saver.exists());
    }

    #[rstest]
    #[should_panic]
    pub fn sqlite_dataset_saver_init_existing(tmp_dir: TempDir) {
        let temp_dir_str = tmp_dir.path().to_str().unwrap().to_string();

        // Create a file in the temp dir
        let _ds_saver1 = SqlDsSaver::new("preprocessed")
            .with_base_dir(temp_dir_str.as_str())
            .with_splits(&["train", "test"])
            .init();

        // Again create a file in the temp dir and it should fail
        let _ds_saver2 = SqlDsSaver::new("preprocessed")
            .with_base_dir(temp_dir_str.as_str())
            .with_splits(&["train", "test"])
            .init();
    }

    #[rstest]
    pub fn sqlite_dataset_saver_init_with_overwrite(tmp_dir: TempDir) {
        let temp_dir_str = tmp_dir.path().to_str().unwrap().to_string();

        // Create a file in the temp dir
        let _ds_saver1 = SqlDsSaver::new("preprocessed")
            .with_base_dir(temp_dir_str.as_str())
            .with_splits(&["train", "test"])
            .init();

        // Again create a file in the temp dir and it should fail
        let _ds_saver2 = SqlDsSaver::new("preprocessed")
            .with_base_dir(temp_dir_str.as_str())
            .with_splits(&["train", "test"])
            .with_overwrite(true)
            .init();
    }

    #[rstest]
    pub fn sqlite_dataset_saver_dataset(dataset_fixture: (SqlDsSaver, TempDir)) {
        // Get the dataset_saver from the fixture and tmp_dir (will be deleted after scope)
        let (dataset_saver, _tmp_dir) = dataset_fixture;

        // Sanity checks of the dataset_saver
        assert!(dataset_saver.exists());
        assert_eq!(dataset_saver.name, "preprocessed");
        assert!(dataset_saver.base_dir.is_some());
        assert!(dataset_saver.is_initialized);
        assert_eq!(
            dataset_saver.splits,
            Some(vec!["train".to_string(), "test".to_string()])
        );
        assert!(dataset_saver.conn_pool.is_some());

        // Make sure that the dataset is empty
        let dataset = dataset_saver.dataset("train");
        assert_eq!(dataset.len(), 0);

        // Insert a sample to "train" split table
        let sample = Complex {
            column_str: "test".to_string(),
            column_bytes: vec![1, 2, 3],
            column_int: 1,
            column_bool: true,
            column_float: 1.0,
            colomn_complex: vec![vec![vec![[1, 2, 3]]]],
        };

        let index = dataset_saver.save("train", &sample);

        // Make sure that the index returned is 0
        assert_eq!(index, 0);

        // Make sure that the dataset has one sample
        let train = dataset_saver.dataset("train");
        assert_eq!(train.len(), 1);

        // Make sure that the sample is the same as the one we inserted
        let sample_out = train.get(0).unwrap();
        assert_eq!(sample, sample_out);

        // Make sure that the dataset returns None for non-existing index
        let non_existing_index = 10;
        let sample_none = train.get(non_existing_index);
        assert!(sample_none.is_none());
    }

    #[rstest]
    #[should_panic]
    pub fn sqlite_dataset_saver_insert_to_wrong_split(dataset_fixture: (SqlDsSaver, TempDir)) {
        // Get the dataset_saver from the fixture and tmp_dir (will be deleted after scope)
        let (dataset_saver, _tmp_dir) = dataset_fixture;

        // Insert a sample to "train" split table
        let sample = Complex {
            column_str: "test".to_string(),
            column_bytes: vec![1, 2, 3],
            column_int: 1,
            column_bool: true,
            column_float: 1.0,
            colomn_complex: vec![vec![vec![[1, 2, 3]]]],
        };

        // Insert to a non-existing split
        dataset_saver.save("non-existing", &sample);
    }

    #[rstest]
    pub fn sqlite_dataset_saver_multi_threads(dataset_fixture: (SqlDsSaver, TempDir)) {
        // Get the dataset_saver from the fixture and tmp_dir (will be deleted after scope)
        let (dataset_saver, _tmp_dir) = dataset_fixture;

        let dataset_saver = Arc::new(dataset_saver);
        let record_count = 200;

        let splits = ["train", "test"];

        (0..record_count).into_par_iter().for_each(|index: i64| {
            let thread_id = std::thread::current().id();
            let sample = Complex {
                column_str: format!("test_{:?}_{}", thread_id, index),
                column_bytes: vec![index as u8, 2, 3],
                column_int: index,
                column_bool: true,
                column_float: 1.0,
                colomn_complex: vec![vec![vec![[1, index as u8, 3]]]],
            };

            // half for train and half for test
            let split = splits[index as usize % 2];

            dataset_saver.save(split, &sample);
        });

        let train = dataset_saver.dataset("train");

        assert_eq!(train.len(), (record_count / 2) as usize);
        let test = dataset_saver.dataset("train");
        assert_eq!(test.len(), (record_count / 2) as usize);
    }
}
