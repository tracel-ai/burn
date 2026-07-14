use std::{
    collections::HashSet,
    fs, io,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use crate::Dataset;

use gix_tempfile::{
    AutoRemove, ContainingDirectory, Handle,
    handle::{Writable, persist},
};
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::{
    SqliteConnectionManager,
    rusqlite::{OpenFlags, OptionalExtension},
};
use sanitize_filename::sanitize;
use serde::{Serialize, de::DeserializeOwned};
use serde_rusqlite::{columns_from_statement, from_row_with_columns};

/// Result type for the sqlite dataset.
pub type Result<T> = core::result::Result<T, SqliteDatasetError>;

/// Sqlite dataset error.
#[derive(thiserror::Error, Debug)]
pub enum SqliteDatasetError {
    /// IO related error.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// Sql related error.
    #[error("Sql error: {0}")]
    Sql(#[from] serde_rusqlite::rusqlite::Error),

    /// Serde related error.
    #[error("Serde error: {0}")]
    Serde(#[from] rmp_serde::encode::Error),

    /// The database file already exists error.
    #[error("Overwrite flag is set to false and the database file already exists: {0}")]
    FileExists(PathBuf),

    /// Error when creating the connection pool.
    #[error("Failed to create connection pool: {0}")]
    ConnectionPool(#[from] r2d2::Error),

    /// Error when persisting the temporary database file.
    #[error("Could not persist the temporary database file: {0}")]
    PersistDbFile(#[from] persist::Error<Writable>),

    /// Any other error.
    #[error("{0}")]
    Other(&'static str),
}

impl From<&'static str> for SqliteDatasetError {
    fn from(s: &'static str) -> Self {
        SqliteDatasetError::Other(s)
    }
}

/// This struct represents a dataset where all items are stored in an SQLite database.
/// Each instance of this struct corresponds to a specific table within the SQLite database,
/// and allows for interaction with the data stored in the table in a structured and typed manner.
///
/// The SQLite database must contain a table with the same name as the `split` field. This table should
/// have a primary key column named `row_id`, which is used to index the rows in the table. The `row_id`
/// should start at 1, while the corresponding dataset `index` should start at 0, i.e., `row_id` = `index` + 1.
///
/// Table columns can be represented in two ways:
///
/// 1. The table can have a column for each field in the `I` struct. In this case, the column names in the table
///    should match the field names of the `I` struct. The field names can be a subset of column names and
///    can be in any order.
///
/// For the supported field types, refer to:
/// - [Serialization field types](https://docs.rs/serde_rusqlite/latest/serde_rusqlite)
/// - [SQLite data types](https://www.sqlite.org/datatype3.html)
///
/// 2. The fields in the `I` struct can be serialized into a single column `item` in the table. In this case, the table
///    should have a single column named `item` of type `BLOB`. This is useful when the `I` struct contains complex fields
///    that cannot be mapped to a SQLite type, such as nested structs, vectors, etc. The serialization is done using
///    [MessagePack](https://msgpack.org/).
///
/// Note: The code automatically figures out which of the above two cases is applicable, and uses the appropriate
/// method to read the data from the table.
#[derive(Debug)]
pub struct SqliteDataset<I> {
    db_file: PathBuf,
    split: String,
    conn_pool: Pool<SqliteConnectionManager>,
    columns: Vec<String>,
    len: usize,
    select_statement: String,
    row_serialized: bool,
    phantom: PhantomData<I>,
}

impl<I> SqliteDataset<I> {
    /// Initializes a `SqliteDataset` from a SQLite database file and a split name.
    pub fn from_db_file<P: AsRef<Path>>(db_file: P, split: &str) -> Result<Self> {
        // Create a connection pool
        let conn_pool = create_conn_pool(&db_file, false)?;

        // Determine how the table is stored
        let row_serialized = Self::check_if_row_serialized(&conn_pool, split)?;

        // Create a select statement and save it
        let select_statement = if row_serialized {
            format!("select item from {split} where row_id = ?")
        } else {
            format!("select * from {split} where row_id = ?")
        };

        // Save the column names and the number of rows
        let (columns, len) = fetch_columns_and_len(&conn_pool, &select_statement, split)?;

        Ok(SqliteDataset {
            db_file: db_file.as_ref().to_path_buf(),
            split: split.to_string(),
            conn_pool,
            columns,
            len,
            select_statement,
            row_serialized,
            phantom: PhantomData,
        })
    }

    /// Returns true if table has two columns: row_id (integer) and item (blob).
    ///
    /// This is used to determine if the table is row serialized or not.
    fn check_if_row_serialized(
        conn_pool: &Pool<SqliteConnectionManager>,
        split: &str,
    ) -> Result<bool> {
        // This struct is used to store the column name and type
        struct Column {
            name: String,
            ty: String,
        }

        const COLUMN_NAME: usize = 1;
        const COLUMN_TYPE: usize = 2;

        let sql_statement = format!("PRAGMA table_info({split})");

        let conn = conn_pool.get()?;

        let mut stmt = conn.prepare(sql_statement.as_str())?;
        let column_iter = stmt.query_map([], |row| {
            Ok(Column {
                name: row
                    .get::<usize, String>(COLUMN_NAME)
                    .unwrap()
                    .to_lowercase(),
                ty: row
                    .get::<usize, String>(COLUMN_TYPE)
                    .unwrap()
                    .to_lowercase(),
            })
        })?;

        let mut columns: Vec<Column> = vec![];

        for column in column_iter {
            columns.push(column?);
        }

        if columns.len() != 2 {
            Ok(false)
        } else {
            // Check if the column names and types match the expected values
            Ok(columns[0].name == "row_id"
                && columns[0].ty == "integer"
                && columns[1].name == "item"
                && columns[1].ty == "blob")
        }
    }

    /// Get the database file name.
    pub fn db_file(&self) -> PathBuf {
        self.db_file.clone()
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
) -> Result<(Vec<String>, usize)> {
    // Save the column names
    let connection = conn_pool.get()?;
    let statement = connection.prepare(select_statement)?;
    let columns = columns_from_statement(&statement);

    // Count the number of rows and save it as len
    //
    // NOTE: Using coalesce(max(row_id), 0) instead of count(*) because count(*) is super slow for large tables.
    // The coalesce(max(row_id), 0) returns 0 if the table is empty, otherwise it returns the max row_id,
    // which corresponds to the number of rows in the table.
    // The main assumption, which always holds true, is that the row_id is always increasing and there are no gaps.
    // This is true for all the datasets that we are using, otherwise row_id will not correspond to the index.
    let mut statement =
        connection.prepare(format!("select coalesce(max(row_id), 0) from {split}").as_str())?;

    let len = statement.query_row([], |row| {
        let len: usize = row.get(0)?;
        Ok(len)
    })?;
    Ok((columns, len))
}

/// Helper function to create a connection pool
fn create_conn_pool<P: AsRef<Path>>(
    db_file: P,
    write: bool,
) -> Result<Pool<SqliteConnectionManager>> {
    let sqlite_flags = if write {
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE
    } else {
        OpenFlags::SQLITE_OPEN_READ_ONLY
    };

    let manager = SqliteConnectionManager::file(db_file).with_flags(sqlite_flags);
    Pool::new(manager).map_err(SqliteDatasetError::ConnectionPool)
}

/// The `SqliteDatasetStorage` struct represents a SQLite database for storing datasets.
/// It consists of an optional name, a database file path, and a base directory for storage.
#[derive(Clone, Debug)]
pub struct SqliteDatasetStorage {
    name: Option<String>,
    db_file: Option<PathBuf>,
    base_dir: Option<PathBuf>,
}

impl SqliteDatasetStorage {
    /// Creates a new instance of `SqliteDatasetStorage` using a dataset name.
    ///
    /// # Arguments
    ///
    /// * `name` - A string slice that holds the name of the dataset.
    pub fn from_name(name: &str) -> Self {
        SqliteDatasetStorage {
            name: Some(name.to_string()),
            db_file: None,
            base_dir: None,
        }
    }

    /// Creates a new instance of `SqliteDatasetStorage` using a database file path.
    ///
    /// # Arguments
    ///
    /// * `db_file` - A reference to the Path that represents the database file path.
    pub fn from_file<P: AsRef<Path>>(db_file: P) -> Self {
        SqliteDatasetStorage {
            name: None,
            db_file: Some(db_file.as_ref().to_path_buf()),
            base_dir: None,
        }
    }

    /// Sets the base directory for storing the dataset.
    ///
    /// # Arguments
    ///
    /// * `base_dir` - A string slice that represents the base directory.
    pub fn with_base_dir<P: AsRef<Path>>(mut self, base_dir: P) -> Self {
        self.base_dir = Some(base_dir.as_ref().to_path_buf());
        self
    }

    /// Checks if the database file exists in the given path.
    ///
    /// # Returns
    ///
    /// * A boolean value indicating whether the file exists or not.
    pub fn exists(&self) -> bool {
        self.db_file().exists()
    }

    /// Fetches the database file path.
    ///
    /// # Returns
    ///
    /// * A `PathBuf` instance representing the file path.
    pub fn db_file(&self) -> PathBuf {
        match &self.db_file {
            Some(db_file) => db_file.clone(),
            None => {
                let name = sanitize(self.name.as_ref().expect("Name is not set"));
                Self::base_dir(self.base_dir.to_owned()).join(format!("{name}.db"))
            }
        }
    }

    /// Determines the base directory for storing the dataset.
    ///
    /// # Arguments
    ///
    /// * `base_dir` - An `Option` that may contain a `PathBuf` instance representing the base directory.
    ///
    /// # Returns
    ///
    /// * A `PathBuf` instance representing the base directory.
    pub fn base_dir(base_dir: Option<PathBuf>) -> PathBuf {
        match base_dir {
            Some(base_dir) => base_dir,
            None => dirs::cache_dir()
                .expect("Could not get cache directory")
                .join("burn-dataset"),
        }
    }

    /// Provides a writer instance for the SQLite dataset.
    ///
    /// # Arguments
    ///
    /// * `overwrite` - A boolean indicating if the existing database file should be overwritten.
    ///
    /// # Returns
    ///
    /// * A `Result` which is `Ok` if the writer could be created, `Err` otherwise.
    pub fn writer<I>(&self, overwrite: bool) -> Result<SqliteDatasetWriter<I>>
    where
        I: Clone + Send + Sync + Serialize + DeserializeOwned,
    {
        SqliteDatasetWriter::new(self.db_file(), overwrite)
    }

    /// Provides a reader instance for the SQLite dataset.
    ///
    /// # Arguments
    ///
    /// * `split` - A string slice that defines the data split for reading (e.g., "train", "test").
    ///
    /// # Returns
    ///
    /// * A `Result` which is `Ok` if the reader could be created, `Err` otherwise.
    pub fn reader<I>(&self, split: &str) -> Result<SqliteDataset<I>>
    where
        I: Clone + Send + Sync + Serialize + DeserializeOwned,
    {
        if !self.exists() {
            panic!("The database file does not exist");
        }

        SqliteDataset::from_db_file(self.db_file(), split)
    }
}

/// This `SqliteDatasetWriter` struct is a SQLite database writer dedicated to storing datasets.
/// It retains the current writer's state and its database connection.
///
/// Being thread-safe, this writer can be concurrently used across multiple threads.
///
/// Typical applications include:
///
/// - Generation of a new dataset
/// - Storage of preprocessed data or metadata
/// - Enlargement of a dataset's item count post preprocessing
#[derive(Debug)]
pub struct SqliteDatasetWriter<I> {
    db_file: PathBuf,
    db_file_tmp: Option<Handle<Writable>>,
    splits: Arc<RwLock<HashSet<String>>>,
    overwrite: bool,
    conn_pool: Option<Pool<SqliteConnectionManager>>,
    is_completed: Arc<RwLock<bool>>,
    phantom: PhantomData<I>,
}

impl<I> SqliteDatasetWriter<I>
where
    I: Clone + Send + Sync + Serialize + DeserializeOwned,
{
    /// Creates a new instance of `SqliteDatasetWriter`.
    ///
    /// # Arguments
    ///
    /// * `db_file` - A reference to the Path that represents the database file path.
    /// * `overwrite` - A boolean indicating if the existing database file should be overwritten.
    ///
    /// # Returns
    ///
    /// * A `Result` which is `Ok` if the writer could be created, `Err` otherwise.
    pub fn new<P: AsRef<Path>>(db_file: P, overwrite: bool) -> Result<Self> {
        let writer = Self {
            db_file: db_file.as_ref().to_path_buf(),
            db_file_tmp: None,
            splits: Arc::new(RwLock::new(HashSet::new())),
            overwrite,
            conn_pool: None,
            is_completed: Arc::new(RwLock::new(false)),
            phantom: PhantomData,
        };

        writer.init()
    }

    /// Initializes the dataset writer by creating the database file, tables, and connection pool.
    ///
    /// # Returns
    ///
    /// * A `Result` which is `Ok` if the writer could be initialized, `Err` otherwise.
    fn init(mut self) -> Result<Self> {
        // Remove the db file if it already exists
        if self.db_file.exists() {
            if self.overwrite {
                fs::remove_file(&self.db_file)?;
            } else {
                return Err(SqliteDatasetError::FileExists(self.db_file));
            }
        }

        // Create the database file directory if it does not exist
        let db_file_dir = self
            .db_file
            .parent()
            .ok_or("Unable to get parent directory")?;

        if !db_file_dir.exists() {
            fs::create_dir_all(db_file_dir)?;
        }

        // Create a temp database file name as {base_dir}/{name}.db.tmp
        let mut db_file_tmp = self.db_file.clone();
        db_file_tmp.set_extension("db.tmp");
        if db_file_tmp.exists() {
            fs::remove_file(&db_file_tmp)?;
        }

        // Create the temp database file and wrap it with a gix_tempfile::Handle
        // This will ensure that the temp file is deleted when the writer is dropped
        // or when process exits with SIGINT or SIGTERM (tempfile crate does not do this)
        gix_tempfile::signal::setup(Default::default());
        self.db_file_tmp = Some(gix_tempfile::writable_at(
            &db_file_tmp,
            ContainingDirectory::Exists,
            AutoRemove::Tempfile,
        )?);

        let conn_pool = create_conn_pool(db_file_tmp, true)?;
        self.conn_pool = Some(conn_pool);

        Ok(self)
    }

    /// Serializes and writes an item to the database. The item is written to the table for the
    /// specified split. If the table does not exist, it is created. If the table exists, the item
    /// is appended to the table. The serialization is done using the [MessagePack](https://msgpack.org/)
    ///
    /// # Arguments
    ///
    /// * `split` - A string slice that defines the data split for writing (e.g., "train", "test").
    /// * `item` - A reference to the item to be written to the database.
    ///
    /// # Returns
    ///
    /// * A `Result` containing the index of the inserted row if successful, an error otherwise.
    pub fn write(&self, split: &str, item: &I) -> Result<usize> {
        // Acquire the read lock (wont't block other reads)
        let is_completed = self.is_completed.read().unwrap();

        // If the writer is completed, return an error
        if *is_completed {
            return Err(SqliteDatasetError::Other(
                "Cannot save to a completed dataset writer",
            ));
        }

        // create the table for the split if it does not exist
        if !self.splits.read().unwrap().contains(split) {
            self.create_table(split)?;
        }

        // Get a connection from the pool
        let conn_pool = self.conn_pool.as_ref().unwrap();
        let conn = conn_pool.get()?;

        // Serialize the item using MessagePack
        let serialized_item = rmp_serde::to_vec(item)?;

        // Turn off the synchronous and journal mode for speed up
        // We are sacrificing durability for speed but it's okay because
        // we always recreate the dataset if it is not completed.
        pragma_update_with_error_handling(&conn, "synchronous", "OFF")?;
        pragma_update_with_error_handling(&conn, "journal_mode", "OFF")?;

        // Insert the serialized item into the database
        let insert_statement = format!("insert into {split} (item) values (?)");
        conn.execute(insert_statement.as_str(), [serialized_item])?;

        // Get the primary key of the last inserted row and convert to index (row_id-1)
        let index = (conn.last_insert_rowid() - 1) as usize;

        Ok(index)
    }

    /// Marks the dataset as completed and persists the temporary database file.
    pub fn set_completed(&mut self) -> Result<()> {
        let mut is_completed = self.is_completed.write().unwrap();

        // Force close the connection pool
        // This is required on Windows platform where the connection pool prevents
        // from persisting the db by renaming the temp file.
        if let Some(pool) = self.conn_pool.take() {
            std::mem::drop(pool);
        }

        // Rename the database file from tmp to db
        let _file_result = self
            .db_file_tmp
            .take() // take ownership of the temporary file and set to None
            .unwrap() // unwrap the temporary file
            .persist(&self.db_file)?
            .ok_or("Unable to persist the database file")?;

        *is_completed = true;
        Ok(())
    }

    /// Creates table for the data split.
    ///
    /// Note: call is idempotent and thread-safe.
    ///
    /// # Arguments
    ///
    /// * `split` - A string slice that defines the data split for the table (e.g., "train", "test").
    ///
    /// # Returns
    ///
    /// * A `Result` which is `Ok` if the table could be created, `Err` otherwise.
    ///
    /// TODO (@antimora): add support creating a table with columns corresponding to the item fields
    fn create_table(&self, split: &str) -> Result<()> {
        // Check if the split already exists
        if self.splits.read().unwrap().contains(split) {
            return Ok(());
        }

        let conn_pool = self.conn_pool.as_ref().unwrap();
        let connection = conn_pool.get()?;
        let create_table_statement = format!(
            "create table if not exists  {split} (row_id integer primary key autoincrement not \
             null, item blob not null)"
        );

        connection.execute(create_table_statement.as_str(), [])?;

        // Add the split to the splits
        self.splits.write().unwrap().insert(split.to_string());

        Ok(())
    }
}

/// Runs a pragma update and ignores the `ExecuteReturnedResults` error.
///
/// Sometimes ExecuteReturnedResults is returned when running a pragma update. This is not an error
/// and can be ignored. This function runs the pragma update and ignores the error if it is
/// `ExecuteReturnedResults`.
fn pragma_update_with_error_handling(
    conn: &PooledConnection<SqliteConnectionManager>,
    setting: &str,
    value: &str,
) -> Result<()> {
    let result = conn.pragma_update(None, setting, value);
    if let Err(error) = result
        && error != rusqlite::Error::ExecuteReturnedResults
    {
        return Err(SqliteDatasetError::Sql(error));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use rayon::prelude::*;
    use rstest::{fixture, rstest};
    use serde::{Deserialize, Serialize};
    use tempfile::{NamedTempFile, TempDir, tempdir};

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
        SqliteDataset::<Sample>::from_db_file("tests/data/sqlite-dataset.db", "train").unwrap()
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
            if let Some(_val) = result {
                match_count += 1
            }
        }

        assert_eq!(match_count, 5);
    }

    #[test]
    fn sqlite_dataset_storage() {
        // Test with non-existing file
        let storage = SqliteDatasetStorage::from_file("non-existing.db");
        assert!(!storage.exists());

        // Test with non-existing name
        let storage = SqliteDatasetStorage::from_name("non-existing.db");
        assert!(!storage.exists());

        // Test with existing file
        let storage = SqliteDatasetStorage::from_file("tests/data/sqlite-dataset.db");
        assert!(storage.exists());
        let result = storage.reader::<Sample>("train");
        assert!(result.is_ok());
        let train = result.unwrap();
        assert_eq!(train.len(), 2);

        // Test get writer
        let temp_file = NamedTempFile::new().unwrap();
        let storage = SqliteDatasetStorage::from_file(temp_file.path());
        assert!(storage.exists());
        let result = storage.writer::<Sample>(true);
        assert!(result.is_ok());
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct Complex {
        column_str: String,
        column_bytes: Vec<u8>,
        column_int: i64,
        column_bool: bool,
        column_float: f64,
        column_complex: Vec<Vec<Vec<[u8; 3]>>>,
    }

    /// Create a temporary directory.
    #[fixture]
    fn tmp_dir() -> TempDir {
        // Create a TempDir. This object will be automatically
        // deleted when it goes out of scope.
        tempdir().unwrap()
    }
    type Writer = SqliteDatasetWriter<Complex>;

    /// Create a SqliteDatasetWriter with a temporary directory.
    /// Make sure to return the temporary directory so that it is not deleted.
    #[fixture]
    fn writer_fixture(tmp_dir: TempDir) -> (Writer, TempDir) {
        let temp_dir_str = tmp_dir.path();
        let storage = SqliteDatasetStorage::from_name("preprocessed").with_base_dir(temp_dir_str);
        let overwrite = true;
        let result = storage.writer::<Complex>(overwrite);
        assert!(result.is_ok());
        let writer = result.unwrap();
        (writer, tmp_dir)
    }

    #[test]
    fn test_new() {
        // Test that the constructor works with overwrite = true
        let test_path = NamedTempFile::new().unwrap();
        let _writer = SqliteDatasetWriter::<Complex>::new(&test_path, true).unwrap();
        assert!(!test_path.path().exists());

        // Test that the constructor works with overwrite = false
        let test_path = NamedTempFile::new().unwrap();
        let result = SqliteDatasetWriter::<Complex>::new(&test_path, false);
        assert!(result.is_err());

        // Test that the constructor works with no existing file
        let temp = NamedTempFile::new().unwrap();
        let test_path = temp.path().to_path_buf();
        assert!(temp.close().is_ok());
        assert!(!test_path.exists());
        let _writer = SqliteDatasetWriter::<Complex>::new(&test_path, true).unwrap();
        assert!(!test_path.exists());
    }

    #[rstest]
    pub fn sqlite_writer_write(writer_fixture: (Writer, TempDir)) {
        // Get the dataset_saver from the fixture and tmp_dir (will be deleted after scope)
        let (writer, _tmp_dir) = writer_fixture;

        assert!(writer.overwrite);
        assert!(!writer.db_file.exists());

        let new_item = Complex {
            column_str: "HI1".to_string(),
            column_bytes: vec![1_u8, 2, 3],
            column_int: 0,
            column_bool: true,
            column_float: 1.0,
            column_complex: vec![vec![vec![[1, 23_u8, 3]]]],
        };

        let index = writer.write("train", &new_item).unwrap();
        assert_eq!(index, 0);

        let mut writer = writer;

        writer.set_completed().expect("Failed to set completed");

        assert!(writer.db_file.exists());
        assert!(writer.db_file_tmp.is_none());

        let result = writer.write("train", &new_item);

        // Should fail because the writer is completed
        assert!(result.is_err());

        let dataset = SqliteDataset::<Complex>::from_db_file(writer.db_file, "train").unwrap();

        let fetched_item = dataset.get(0).unwrap();
        assert_eq!(fetched_item, new_item);
        assert_eq!(dataset.len(), 1);
    }

    #[rstest]
    pub fn sqlite_writer_write_multi_thread(writer_fixture: (Writer, TempDir)) {
        // Get the dataset_saver from the fixture and tmp_dir (will be deleted after scope)
        let (writer, _tmp_dir) = writer_fixture;

        let writer = Arc::new(writer);
        let record_count = 20;

        let splits = ["train", "test"];

        (0..record_count).into_par_iter().for_each(|index: i64| {
            let thread_id: std::thread::ThreadId = std::thread::current().id();
            let sample = Complex {
                column_str: format!("test_{thread_id:?}_{index}"),
                column_bytes: vec![index as u8, 2, 3],
                column_int: index,
                column_bool: true,
                column_float: 1.0,
                column_complex: vec![vec![vec![[1, index as u8, 3]]]],
            };

            // half for train and half for test
            let split = splits[index as usize % 2];

            let _index = writer.write(split, &sample).unwrap();
        });

        let mut writer = Arc::try_unwrap(writer).unwrap();

        writer
            .set_completed()
            .expect("Should set completed successfully");

        let train =
            SqliteDataset::<Complex>::from_db_file(writer.db_file.clone(), "train").unwrap();
        let test = SqliteDataset::<Complex>::from_db_file(writer.db_file, "test").unwrap();

        assert_eq!(train.len(), record_count as usize / 2);
        assert_eq!(test.len(), record_count as usize / 2);
    }
}
