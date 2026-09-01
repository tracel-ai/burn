use std::{
    collections::HashSet,
    ffi::OsString,
    fs, io,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, PoisonError, RwLock},
};

use crate::Dataset;

// Turso exposes an async API, but for a local database file every `poll` runs its IO inline
// before returning, so no reactor is involved and a parking executor is enough to drive it.
// `Dataset` is a synchronous trait, and this is where the two meet.
use futures_lite::future::block_on;
use gix_tempfile::{
    AutoRemove, ContainingDirectory, Handle,
    handle::{Writable, persist},
};
use r2d2::Pool;
use sanitize_filename::sanitize;
use serde::{Serialize, de::DeserializeOwned};
use turso::{Builder, Connection, Database, Value};

mod de;

use de::from_row_with_columns;
// Re-exported because `SqliteDatasetError::Row` carries it: a payload users cannot name is one
// they cannot match on or wrap.
pub use de::RowError;

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
    Sql(#[from] turso::Error),

    /// Serde related error.
    #[error("Serde error: {0}")]
    Serde(#[from] rmp_serde::encode::Error),

    /// Deserialization error when reading a row.
    #[error("Deserialize error: {0}")]
    Deserialize(#[from] rmp_serde::decode::Error),

    /// Row deserialization error.
    #[error("Row error: {0}")]
    Row(#[from] RowError),

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
/// The database is read and written with [Turso](https://turso.tech/), which implements the engine
/// in Rust rather than linking libsqlite3, so the files stay interchangeable with any other SQLite
/// tooling. Note that this does not make the build free of C: turso pulls `simsimd` for its SIMD
/// kernels, which compiles C unconditionally.
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
/// For the supported field types, refer to
/// [SQLite data types](https://www.sqlite.org/datatype3.html). Each column is handed to serde as
/// the value its storage class implies: `INTEGER` as `i64`, `REAL` as `f64`, `TEXT` as `String`,
/// and `BLOB` as a byte sequence. Booleans are read back from integers, and a nullable column maps
/// to an `Option` field. Reading a NULL into a bare `f32` or `f64` field yields `NaN`.
///
/// A `Vec<u8>` field derives as a sequence, so a large `BLOB` column is walked one byte at a time.
/// Tagging the field `#[serde(with = "serde_bytes")]` moves the blob across instead, which is
/// substantially cheaper for columns of any size.
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
    conn_pool: Pool<TursoConnectionManager>,
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
        let database = open_database(&db_file, false)?;
        let conn_pool = Pool::new(TursoConnectionManager { database })?;

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
        conn_pool: &Pool<TursoConnectionManager>,
        split: &str,
    ) -> Result<bool> {
        let conn = conn_pool.get()?;

        // A prepared statement already reports each column's name and declared type, so the table
        // does not need to be described separately.
        let columns = block_on(conn.prepare(&format!("select * from {split}")))?.columns();

        let matches = |index: usize, name: &str, ty: &str| {
            columns[index].name().eq_ignore_ascii_case(name)
                && columns[index]
                    .decl_type()
                    .is_some_and(|declared| declared.eq_ignore_ascii_case(ty))
        };

        Ok(columns.len() == 2 && matches(0, "row_id", "integer") && matches(1, "item", "blob"))
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

impl<I: DeserializeOwned> SqliteDataset<I> {
    /// Decodes one row, in whichever of the two table layouts this dataset uses.
    fn item_from_row(&self, row: &turso::Row) -> Result<I> {
        if self.row_serialized {
            // A single column `item`, serialized with MessagePack
            match row.get_value(0)? {
                Value::Blob(blob) => Ok(rmp_serde::from_slice::<I>(&blob)?),
                _ => Err(SqliteDatasetError::Other("expected a blob column")),
            }
        } else {
            // A column per field, decoded column by column
            Ok(from_row_with_columns::<I>(row, &self.columns)?)
        }
    }
}

impl<I> Dataset<I, SqliteDatasetError> for SqliteDataset<I>
where
    I: Clone + Send + Sync + DeserializeOwned,
{
    /// Get an item from the dataset.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len()`.
    fn get(&self, index: usize) -> Result<I> {
        assert!(
            index < self.len,
            "Index out of bounds for SqliteDataset: {} >= {}",
            index,
            self.len,
        );

        // Row ids start with 1 (one) and index starts with 0 (zero).
        // Turso binds integers as `i64`, so `usize` is not a valid parameter type.
        let row_id = (index + 1) as i64;

        // Get a connection from the pool
        let connection = self.conn_pool.get()?;

        let row = block_on(async {
            // `prepare_cached` keeps the compiled program on the pooled connection, which is
            // what makes repeated single-row reads cheap. `query_row` runs the statement to
            // completion, which is what keeps dropping it from rolling back an open transaction.
            let mut statement = connection.prepare_cached(&self.select_statement).await?;
            statement.query_row([row_id]).await
        })
        .map_err(|error| match error {
            // `len` comes from `max(row_id)` and assumes ids are contiguous, so a gap shows up
            // here rather than at construction. Say that instead of reporting a database fault.
            turso::Error::QueryReturnedNoRows => SqliteDatasetError::Other(
                "no row for this index; row_id values must be contiguous and start at 1",
            ),
            error => error.into(),
        })?;

        self.item_from_row(&row)
    }

    /// Get multiple items from the dataset in a single query.
    ///
    /// # Panics
    ///
    /// Panics if `indexes[i] >= len()`.
    fn get_many(&self, indexes: Vec<usize>) -> Result<Vec<I>> {
        if indexes.is_empty() {
            return Ok(Vec::new());
        }

        for &index in &indexes {
            assert!(
                index < self.len,
                "Index out of bounds for SqliteDataset: {} >= {}",
                index,
                self.len,
            );
        }

        // Bind (row_id, ord) pairs through a CTE so the engine returns the rows already ordered
        // (and duplicated) to match the requested `indexes`, instead of reordering in Rust.
        let values_clause = vec!["(?, ?)"; indexes.len()].join(", ");
        let params: Vec<i64> = indexes
            .iter()
            .enumerate()
            .flat_map(|(ord, &index)| [(index + 1) as i64, ord as i64])
            .collect();

        let split = &self.split;
        let connection = self.conn_pool.get()?;

        let selection = if self.row_serialized { "t.item" } else { "t.*" };
        let query = format!(
            "WITH req(row_id, ord) AS (VALUES {values_clause}) \
             SELECT {selection} FROM req JOIN {split} t ON t.row_id = req.row_id ORDER BY req.ord"
        );

        // Deserialize as rows arrive rather than collecting them first, so a batch never holds
        // both the raw rows and the decoded items at once.
        //
        // `prepare` rather than `prepare_cached`: the statement carries one `(?, ?)` pair per
        // requested index, so every distinct batch size is a distinct cache key, and turso's
        // per-connection cache never evicts. A caller varying its batch size would pin a compiled
        // program and its SQL text for every size it ever used, on every pooled connection.
        block_on(async {
            let mut statement = connection.prepare(&query).await?;
            let mut rows = statement.query(params).await?;

            let mut items = Vec::with_capacity(indexes.len());
            while let Some(row) = rows.next().await? {
                items.push(self.item_from_row(&row)?);
            }

            // The join drops requested ids that do not exist, so a gapped table would otherwise
            // hand back a short batch as success, where `get` on the same index errors.
            if items.len() != indexes.len() {
                return Err(SqliteDatasetError::Other(
                    "fewer rows than indexes requested; row_id values must be contiguous and \
                     start at 1",
                ));
            }

            Ok(items)
        })
    }

    /// Return the number of rows in the dataset.
    fn len(&self) -> usize {
        self.len
    }
}

/// Fetch the column names and the number of rows from the database.
fn fetch_columns_and_len(
    conn_pool: &Pool<TursoConnectionManager>,
    select_statement: &str,
    split: &str,
) -> Result<(Vec<String>, usize)> {
    let connection = conn_pool.get()?;

    let (columns, max_row_id) = block_on(async {
        // Save the column names
        let statement = connection.prepare(select_statement).await?;
        let columns = statement.column_names();

        // Count the number of rows and save it as len
        //
        // NOTE: Using max(row_id) instead of count(*) because count(*) walks the table where
        // max(row_id) is a single backward seek. The max row_id corresponds to the number of rows
        // in the table.
        // The main assumption, which always holds true, is that the row_id is always increasing and there are no gaps.
        // This is true for all the datasets that we are using, otherwise row_id will not correspond to the index.
        //
        // The aggregate has to stand alone for turso to take that fast path, so an empty table's
        // NULL is turned into 0 here rather than by wrapping the call in `coalesce`. Note that
        // `EXPLAIN QUERY PLAN` reports a scan for both forms, so the difference only shows up in
        // the emitted bytecode, or on a clock.
        let mut statement = connection
            .prepare(format!("select max(row_id) from {split}").as_str())
            .await?;
        let max_row_id = statement.query_row(()).await?.get_value(0)?;

        Ok::<_, turso::Error>((columns, max_row_id))
    })?;

    // Only NULL means "empty table". SQLite's dynamic typing lets a `row_id` column hold text or
    // a float, and a dataset indexed by something that is not an integer is not one this can read
    // by row id, so say so rather than reporting it as empty.
    let len = match max_row_id {
        Value::Null => 0,
        Value::Integer(max_row_id) => usize::try_from(max_row_id).map_err(|_| {
            SqliteDatasetError::Other("row_id is negative, so it cannot index the dataset")
        })?,
        _ => {
            return Err(SqliteDatasetError::Other(
                "row_id is not an integer, so it cannot index the dataset",
            ));
        }
    };

    Ok((columns, len))
}

/// Hands out connections to a single turso database.
///
/// Pooling is not just an optimization here: a turso [`Connection`] rejects concurrent use from
/// more than one thread at a time, so a multi-threaded reader needs one connection per caller.
#[derive(Debug)]
struct TursoConnectionManager {
    database: Database,
}

impl r2d2::ManageConnection for TursoConnectionManager {
    type Connection = Connection;
    type Error = turso::Error;

    fn connect(&self) -> core::result::Result<Connection, turso::Error> {
        self.database.connect()
    }

    fn is_valid(&self, _conn: &mut Connection) -> core::result::Result<(), turso::Error> {
        Ok(())
    }

    fn has_broken(&self, _conn: &mut Connection) -> bool {
        false
    }
}

/// Opens a database file. Reads and writes go through different shapes, so `write` picks which:
/// the reader fans a pool out over one handle, the writer keeps a single connection.
fn open_database<P: AsRef<Path>>(db_file: P, write: bool) -> Result<Database> {
    // Turso takes the path as `&str`, and `to_string_lossy` would substitute U+FFFD without
    // saying so, pointing the engine at a different file than the caller named. In the writer
    // that is silent total loss: rows land in the mangled file while the untouched original is
    // the one published.
    let db_file = db_file.as_ref().to_str().ok_or(SqliteDatasetError::Other(
        "database path is not valid UTF-8, which the turso engine requires",
    ))?;

    // The exact version pin on turso is load-bearing here beyond the public error type: read-only
    // enforcement is behaviour of a prerelease, covered by no stability promise.
    Ok(block_on(
        Builder::new_local(db_file).read_only(!write).build(),
    )?)
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
    overwrite: bool,
    state: Option<Mutex<WriteState>>,
    is_completed: Arc<RwLock<bool>>,
    phantom: PhantomData<I>,
}

/// Number of rows batched into a single transaction.
///
/// Turso is WAL-only and ignores `journal_mode = OFF`, so an autocommit insert pays for a WAL
/// commit per row. Batching amortizes that away.
///
/// A batch that never commits costs more than the rows in it. [`SqliteDatasetWriter::write`] has
/// already handed their indexes to the caller, and turso reuses the rowids once the transaction is
/// discarded, so those indexes would go on to name different items. That is why a rolled-back
/// batch makes the whole dataset unpublishable rather than merely shorter: the only safe reading
/// of an index is one from a run that reached [`SqliteDatasetWriter::set_completed`] successfully.
const WRITE_BATCH_SIZE: usize = 256;

/// The writer's single connection and the batch currently in flight on it.
///
/// One connection rather than a pool, because a database admits one writer at a time anyway and a
/// turso [`Connection`] forbids concurrent use. Callers serialize on the enclosing mutex, which is
/// held only for the insert itself: item serialization happens before it is taken.
#[derive(Debug)]
struct WriteState {
    connection: Connection,
    /// Zero means no transaction is open. Bumped as soon as `BEGIN` succeeds rather than after the
    /// insert, so that a failing insert cannot strand a transaction nothing would ever close.
    pending: usize,
    /// Set once a batch has been rolled back. The rows it held are gone, so the dataset can no
    /// longer be published: without this, the commit that failed would be a no-op on the retry and
    /// `set_completed` would report success over a dataset missing up to a full batch.
    lost_a_batch: bool,
    /// Splits whose table has already been created on this connection.
    splits: HashSet<String>,
}

impl WriteState {
    /// Inserts one serialized item, opening a batch transaction if none is in flight, and returns
    /// the index of the new row.
    fn insert(&mut self, split: &str, item: Vec<u8>) -> Result<usize> {
        block_on(async {
            if self.pending == 0 {
                self.connection.execute("BEGIN", ()).await?;
            }
            self.pending += 1;

            let insert_statement = format!("insert into {split} (item) values (?)");
            let mut statement = self.connection.prepare_cached(&insert_statement).await?;
            statement.execute([item]).await
        })?;

        // Get the primary key of the last inserted row and convert to index (row_id-1)
        let index = (self.connection.last_insert_rowid() - 1) as usize;

        if self.pending >= WRITE_BATCH_SIZE {
            self.commit()?;
        }

        Ok(index)
    }

    /// Commits the batch in flight, if there is one.
    fn commit(&mut self) -> Result<()> {
        if self.pending > 0 {
            let committed = block_on(self.connection.execute("COMMIT", ()));

            if committed.is_err() {
                // A refused `COMMIT` can leave the transaction open. Roll it back so that
                // `pending == 0` keeps meaning "nothing in flight": left as it was, the next
                // insert would skip its `BEGIN` and run against a transaction it does not know
                // about. The rows in it are gone, which is what `lost_a_batch` records.
                let _ = block_on(self.connection.execute("ROLLBACK", ()));
                self.lost_a_batch = true;
            }

            self.pending = 0;
            committed?;
        }
        Ok(())
    }

    /// Folds the write-ahead log back into the database file.
    ///
    /// The `turso` crate exposes no `close()`, and dropping a connection does not checkpoint, so
    /// the tail of the dataset lives in a `-wal` sidecar until this runs. Publishing the database
    /// file without it would hand over a dataset missing everything not yet backfilled.
    ///
    /// Failure arrives as a result row rather than an error: column 0 is non-zero when the
    /// checkpoint did not complete. Turso funnels every cause into that one flag - a reader
    /// holding an open transaction, a full disk, an IO error, a corrupt header - and keeps the
    /// real reason to a `tracing::debug!` of its own, which burn-dataset's `tracing` feature
    /// surfaces. Taking the pragma's `Ok` at face value would publish a truncated dataset.
    ///
    /// The remaining columns carry nothing to check: a TRUNCATE checkpoint zeroes them on the way
    /// out, so it reports `(0, 0, 0)` however many frames it folded in.
    fn checkpoint(&self) -> Result<()> {
        if self.lost_a_batch {
            return Err(SqliteDatasetError::Other(
                "A batch was rolled back earlier, so this dataset is missing rows and will not be \
                 published. Rebuild it from scratch",
            ));
        }

        let mut failed = false;
        block_on(
            self.connection
                .pragma_query("wal_checkpoint(TRUNCATE)", |row| {
                    failed |= !matches!(row.get_value(0), Ok(Value::Integer(0)));
                    Ok(())
                }),
        )?;

        if failed {
            return Err(SqliteDatasetError::Other(
                "Could not checkpoint the write-ahead log, so the database was left unpublished \
                 rather than published incomplete. Turso does not report why; enable the \
                 `tracing` feature and log at debug level to see the underlying cause",
            ));
        }

        Ok(())
    }

    /// Creates the table backing a split, once per split.
    ///
    /// The batch in flight is committed first so that the schema change never rides along inside a
    /// transaction that is still accumulating rows.
    fn create_table(&mut self, split: &str) -> Result<()> {
        if self.splits.contains(split) {
            return Ok(());
        }

        self.commit()?;
        let create_table_statement = format!(
            "create table if not exists  {split} (row_id integer primary key autoincrement not \
             null, item blob not null)"
        );
        block_on(self.connection.execute(&create_table_statement, ()))?;
        self.splits.insert(split.to_string());
        Ok(())
    }
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
            overwrite,
            state: None,
            is_completed: Arc::new(RwLock::new(false)),
            phantom: PhantomData,
        };

        writer.init()
    }

    /// Initializes the dataset writer by creating the database file, tables, and connection.
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
                return Err(SqliteDatasetError::FileExists(self.db_file.clone()));
            }
        }

        // An earlier database may have left a `-wal` sidecar beside this path, with or without the
        // database itself still there. Whoever opens the file this writer publishes would apply
        // that sidecar as if it belonged to it.
        remove_wal_file(&self.db_file)?;

        // Create the database file directory if it does not exist
        let db_file_dir = self
            .db_file
            .parent()
            .ok_or("Unable to get parent directory")?;

        if !db_file_dir.exists() {
            fs::create_dir_all(db_file_dir)?;
        }

        // Create a temp database file name as {base_dir}/{name}.db.tmp
        let db_file_tmp = tmp_db_file(&self.db_file);
        if db_file_tmp.exists() {
            fs::remove_file(&db_file_tmp)?;
        }

        // Turso writes through a `-wal` sidecar, which a previous aborted run may have left
        // behind. Starting from a stale one would corrupt the fresh database.
        remove_wal_file(&db_file_tmp)?;

        // Create the temp database file and wrap it with a gix_tempfile::Handle
        // This will ensure that the temp file is deleted when the writer is dropped
        // or when process exits with SIGINT or SIGTERM (tempfile crate does not do this)
        gix_tempfile::signal::setup(Default::default());
        self.db_file_tmp = Some(gix_tempfile::writable_at(
            &db_file_tmp,
            ContainingDirectory::Exists,
            AutoRemove::Tempfile,
        )?);

        // The connection keeps the database alive, so the handle does not need to be held on to.
        let connection = open_database(&db_file_tmp, true)?.connect()?;
        self.state = Some(Mutex::new(WriteState {
            connection,
            pending: 0,
            lost_a_batch: false,
            splits: HashSet::new(),
        }));

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
    ///
    /// The index is provisional until [`Self::set_completed`] returns `Ok`. It belongs to a row in
    /// a transaction that has not committed yet, and a batch that is later rolled back gives its
    /// rowids back to the engine. A run whose `set_completed` fails publishes nothing, so indexes
    /// recorded during it must be discarded along with it.
    pub fn write(&self, split: &str, item: &I) -> Result<usize> {
        // Acquire the read lock (wont't block other reads)
        let is_completed = self
            .is_completed
            .read()
            .unwrap_or_else(PoisonError::into_inner);

        // If the writer is completed, return an error
        if *is_completed {
            return Err(SqliteDatasetError::Other(
                "Cannot save to a completed dataset writer",
            ));
        }

        // Serialize the item using MessagePack, before taking the connection mutex so that this
        // part stays parallel across threads
        let serialized_item = rmp_serde::to_vec(item)?;

        let state = self.state.as_ref().ok_or(SqliteDatasetError::Other(
            "Cannot save to a completed dataset writer",
        ))?;
        let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);

        // create the table for the split if it does not exist
        state.create_table(split)?;
        state.insert(split, serialized_item)
    }

    /// Marks the dataset as completed and persists the temporary database file.
    pub fn set_completed(&mut self) -> Result<()> {
        let mut is_completed = self
            .is_completed
            .write()
            .unwrap_or_else(PoisonError::into_inner);

        // Commit the batch still in flight and fold the write-ahead log back into the database
        // file, so that the file persisted below actually holds the dataset.
        //
        // The state stays in place until both succeed. Taking it up front would make a failure
        // unrecoverable in the worst way: the retry would find nothing left to commit, fall
        // through to the rename below, and publish the very uncheckpointed database that
        // `checkpoint` had just refused to publish.
        {
            let state = self
                .state
                .as_mut()
                .ok_or(SqliteDatasetError::Other(
                    "Cannot complete a dataset writer twice",
                ))?
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner);
            state.commit()?;
            state.checkpoint()?;
        }

        // Dropping the connection is required on the Windows platform, where an open connection
        // prevents persisting the db by renaming the temp file.
        self.state = None;

        // Rename the database file from tmp to db
        let _file_result = self
            .db_file_tmp
            .take() // take ownership of the temporary file and set to None
            .unwrap() // unwrap the temporary file
            .persist(&self.db_file)?
            .ok_or("Unable to persist the database file")?;

        // The checkpoint above emptied the sidecar, but turso leaves the file itself behind.
        // Removing it is best effort: the dataset is already published at this point, so failing
        // to clean up a zero-length leftover must not fail the write.
        let _ = fs::remove_file(wal_file(&tmp_db_file(&self.db_file)));

        *is_completed = true;
        Ok(())
    }
}

impl<I> Drop for SqliteDatasetWriter<I> {
    /// Removes the write-ahead log beside the temporary database.
    ///
    /// `gix_tempfile` owns `{name}.db.tmp`, but it knows nothing about the `-wal` sidecar turso
    /// writes next to it. That sidecar holds the tail of the dataset, so a writer abandoned before
    /// `set_completed` would otherwise leave the bulk of what it wrote on disk with nothing left
    /// to reference it.
    ///
    /// This covers an ordinary drop only. `gix_tempfile::signal::setup` still takes the `.db.tmp`
    /// away on SIGINT or SIGTERM, but `Drop` does not run there, so a signalled process leaves the
    /// sidecar behind; `init` clears it on the next run.
    fn drop(&mut self) {
        // The connection goes first: it holds the sidecar open until it does.
        self.state = None;
        let _ = fs::remove_file(wal_file(&tmp_db_file(&self.db_file)));
    }
}

/// Path of the temporary database the writer fills in before publishing it as `db_file`.
fn tmp_db_file(db_file: &Path) -> PathBuf {
    let mut db_file_tmp = db_file.to_path_buf();
    db_file_tmp.set_extension("db.tmp");
    db_file_tmp
}

/// Path of the write-ahead log that turso keeps beside `db_file`.
fn wal_file(db_file: &Path) -> PathBuf {
    let mut wal_file = OsString::from(db_file);
    wal_file.push("-wal");
    PathBuf::from(wal_file)
}

/// Removes the write-ahead log beside `db_file`, if there is one.
fn remove_wal_file(db_file: &Path) -> Result<()> {
    let wal_file = wal_file(db_file);
    if wal_file.exists() {
        fs::remove_file(wal_file)?;
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
    #[should_panic(expected = "Index out of bounds")]
    pub fn get_none(train_dataset: SqlDs) {
        train_dataset.get(10).unwrap();
    }

    #[rstest]
    pub fn get_many_out_of_order_with_duplicates(train_dataset: SqlDs) {
        // Requested out of order and with a duplicate; result must follow the requested order.
        let items = train_dataset.get_many(vec![1, 0, 1]).unwrap();

        assert_eq!(items.len(), 3);
        assert_eq!(items[0], train_dataset.get(1).unwrap());
        assert_eq!(items[1], train_dataset.get(0).unwrap());
        assert_eq!(items[2], train_dataset.get(1).unwrap());
    }

    #[rstest]
    pub fn get_many_empty(train_dataset: SqlDs) {
        assert_eq!(train_dataset.get_many(vec![]).unwrap(), Vec::new());
    }

    #[rstest]
    #[should_panic(expected = "Index out of bounds")]
    pub fn get_many_out_of_bounds(train_dataset: SqlDs) {
        train_dataset.get_many(vec![0, 10]).unwrap();
    }

    #[rstest]
    pub fn multi_thread(train_dataset: SqlDs) {
        let dataset_len = train_dataset.len();
        let indices: Vec<usize> = vec![0, 1, 1, 3, 4, 5, 6, 0, 8, 1];
        let valid_indices: Vec<usize> = indices.into_iter().filter(|&i| i < dataset_len).collect();

        let results: Vec<Sample> = valid_indices
            .par_iter()
            .map(|&i| train_dataset.get(i).unwrap())
            .collect();

        assert_eq!(results.len(), 5);
    }

    /// Reading must leave the database file exactly as it was found.
    ///
    /// Turso writes through a `-wal` sidecar and rewrites the file header into WAL mode as soon as
    /// it is opened for writing, so a reader that is not genuinely read-only would silently modify
    /// the dataset it was handed and litter the directory next to it.
    #[rstest]
    pub fn reading_leaves_the_database_file_untouched(tmp_dir: TempDir) {
        let db_file = tmp_dir.path().join("sqlite-dataset.db");
        fs::copy("tests/data/sqlite-dataset.db", &db_file).unwrap();
        let before = fs::read(&db_file).unwrap();

        let dataset = SqliteDataset::<Sample>::from_db_file(&db_file, "train").unwrap();
        assert_eq!(dataset.len(), 2);
        dataset.get(0).unwrap();
        dataset.get_many(vec![0, 1]).unwrap();
        drop(dataset);

        assert_eq!(fs::read(&db_file).unwrap(), before);
        assert!(!wal_file(&db_file).exists());
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct Typed {
        #[serde(rename = "MedInc")]
        median_income: f32,
        count: usize,
        ratio: f64,
        label: Option<String>,
        missing: Option<f64>,
    }

    /// Covers the column shapes a HuggingFace import produces, which the row-serialized path never
    /// reaches: renamed fields, floats read from both `REAL` and `INTEGER` storage, and `NULL`
    /// columns read into `Option`.
    #[rstest]
    pub fn get_maps_columns_onto_typed_fields(tmp_dir: TempDir) {
        let db_file = tmp_dir.path().join("typed.db");

        {
            let connection = open_database(&db_file, true).unwrap().connect().unwrap();
            block_on(async {
                connection
                    .execute(
                        "create table train (\"MedInc\" REAL, count INTEGER, ratio REAL, \
                         label TEXT, missing REAL, row_id INTEGER NOT NULL, PRIMARY KEY (row_id))",
                        (),
                    )
                    .await?;
                // `ratio` holds an integer in the first row even though the column is declared
                // REAL, which SQLite's dynamic typing allows and real imports do produce.
                connection
                    .execute(
                        "insert into train values (8.3252, 41, 2, 'first', NULL, 1)",
                        (),
                    )
                    .await?;
                connection
                    .execute(
                        "insert into train values (8.3014, 21, 0.5, NULL, 1.5, 2)",
                        (),
                    )
                    .await?;
                // A NULL read into a bare float, which the public docs promise arrives as NaN
                // rather than as an error.
                connection
                    .execute(
                        "insert into train values (NULL, 7, 1.0, 'third', NULL, 3)",
                        (),
                    )
                    .await?;
                connection
                    .pragma_query("wal_checkpoint(TRUNCATE)", |_| Ok(()))
                    .await
            })
            .unwrap();
        }
        fs::remove_file(wal_file(&db_file)).unwrap();

        let dataset = SqliteDataset::<Typed>::from_db_file(&db_file, "train").unwrap();
        assert_eq!(dataset.len(), 3);
        assert_eq!(
            dataset.get(0).unwrap(),
            Typed {
                median_income: 8.3252,
                count: 41,
                ratio: 2.0,
                label: Some("first".to_string()),
                missing: None,
            }
        );
        assert_eq!(
            dataset.get(1).unwrap(),
            Typed {
                median_income: 8.3014,
                count: 21,
                ratio: 0.5,
                label: None,
                missing: Some(1.5),
            }
        );

        // NULL into a bare `f32` yields NaN, matching what serde_rusqlite did.
        let third = dataset.get(2).unwrap();
        assert!(third.median_income.is_nan());
        assert_eq!(third.count, 7);
        assert_eq!(third.label, Some("third".to_string()));
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

        let dataset = SqliteDataset::<Complex>::from_db_file(&writer.db_file, "train").unwrap();

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
        let test = SqliteDataset::<Complex>::from_db_file(&writer.db_file, "test").unwrap();

        assert_eq!(train.len(), record_count as usize / 2);
        assert_eq!(test.len(), record_count as usize / 2);
    }

    /// A rolled-back batch must make the dataset permanently unpublishable.
    ///
    /// The rows are gone and their indexes have already been handed to the caller, so publishing
    /// anything afterwards would hand over a dataset those indexes no longer describe.
    #[rstest]
    pub fn a_lost_batch_can_never_be_published(writer_fixture: (Writer, TempDir)) {
        let (writer, _tmp_dir) = writer_fixture;
        let item = Complex {
            column_str: "HI".to_string(),
            column_bytes: vec![1, 2, 3],
            column_int: 0,
            column_bool: true,
            column_float: 1.0,
            column_complex: vec![vec![vec![[1, 2, 3]]]],
        };
        writer.write("train", &item).unwrap();

        // Stand in for a refused COMMIT, which `commit` records this way after rolling back.
        writer.state.as_ref().unwrap().lock().unwrap().lost_a_batch = true;

        let mut writer = writer;
        assert!(writer.set_completed().is_err(), "published a lost batch");
        assert!(!writer.db_file.exists());

        // Still refused on a retry, and after further successful writes.
        writer.write("train", &item).unwrap();
        assert!(writer.set_completed().is_err(), "published a lost batch");
        assert!(!writer.db_file.exists());
    }

    /// `get_many` binds two parameters per index, so a large batch must not run into a
    /// bind-variable limit.
    ///
    /// Turso 0.8.0-pre.7 enforces none (verified well past SQLite's own 32766), but the engine is
    /// pinned at a prerelease and the dataloader lets callers pick any batch size, so this pins
    /// the behaviour rather than trusting it.
    #[rstest]
    pub fn get_many_handles_batches_past_the_sqlite_variable_limit(tmp_dir: TempDir) {
        let db_file = tmp_dir.path().join("wide.db");
        let item = |index: usize| Complex {
            column_str: format!("item_{index}"),
            column_bytes: vec![index as u8, 2, 3],
            column_int: index as i64,
            column_bool: true,
            column_float: 1.0,
            column_complex: vec![vec![vec![[1, index as u8, 3]]]],
        };

        // 1200 indexes is 2400 bind parameters, comfortably past the historical 999.
        let count = 1200;
        let mut writer = SqliteDatasetWriter::<Complex>::new(&db_file, true).unwrap();
        for index in 0..count {
            writer.write("train", &item(index)).unwrap();
        }
        writer.set_completed().unwrap();

        let dataset = SqliteDataset::<Complex>::from_db_file(&db_file, "train").unwrap();
        let items = dataset.get_many((0..count).collect()).unwrap();
        assert_eq!(items.len(), count);
        for (index, got) in items.iter().enumerate() {
            assert_eq!(got.column_int, index as i64);
        }
    }

    /// A `row_id` that is not an integer cannot index the dataset, and must not read as empty.
    ///
    /// SQLite's dynamic typing allows it, and `max()` over such a column returns text. Reporting
    /// `len() == 0` would make a populated dataset look empty and silently train on nothing.
    #[rstest]
    pub fn non_integer_row_id_is_an_error(tmp_dir: TempDir) {
        let db_file = tmp_dir.path().join("textual.db");

        {
            let connection = open_database(&db_file, true).unwrap().connect().unwrap();
            block_on(async {
                connection
                    .execute("create table train (row_id TEXT, name TEXT)", ())
                    .await?;
                connection
                    .execute("insert into train values ('a', 'first')", ())
                    .await?;
                connection
                    .pragma_query("wal_checkpoint(TRUNCATE)", |_| Ok(()))
                    .await
            })
            .unwrap();
        }
        fs::remove_file(wal_file(&db_file)).unwrap();

        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct Named {
            name: String,
        }

        let result = SqliteDataset::<Named>::from_db_file(&db_file, "train");
        assert!(
            result.is_err(),
            "a textual row_id should be rejected, not reported as an empty dataset"
        );
    }

    /// A writer abandoned without `set_completed` must not leave its write-ahead log behind.
    ///
    /// `gix_tempfile` removes the `.db.tmp` file itself, but the sidecar holds the tail of the
    /// dataset and is nobody else's responsibility.
    #[rstest]
    pub fn sqlite_writer_abandoned_leaves_nothing_behind(tmp_dir: TempDir) {
        let db_file = tmp_dir.path().join("abandoned.db");
        let item = Complex {
            column_str: "HI".to_string(),
            column_bytes: vec![1, 2, 3],
            column_int: 0,
            column_bool: true,
            column_float: 1.0,
            column_complex: vec![vec![vec![[1, 2, 3]]]],
        };

        {
            let writer = SqliteDatasetWriter::<Complex>::new(&db_file, true).unwrap();
            for _ in 0..1000 {
                writer.write("train", &item).unwrap();
            }
            // Dropped without set_completed, as an interrupted preprocessing job would be.
        }

        let db_file_tmp = tmp_db_file(&db_file);
        assert!(!db_file_tmp.exists(), "temporary database left behind");
        assert!(
            !wal_file(&db_file_tmp).exists(),
            "abandoned writer leaked its write-ahead log"
        );
        assert!(!db_file.exists(), "nothing should have been published");
    }

    /// Reads have to come from a pool: a turso connection rejects concurrent use outright, so a
    /// shared one fails with `Misuse("concurrent use forbidden")` rather than merely serializing.
    /// The `multi_thread` test above is far too small to ever schedule two threads at once.
    #[rstest]
    pub fn concurrent_reads_scale_across_the_pool(tmp_dir: TempDir) {
        let db_file = tmp_dir.path().join("concurrent.db");
        let item = |index: usize| Complex {
            column_str: format!("item_{index}"),
            column_bytes: vec![index as u8, 2, 3],
            column_int: index as i64,
            column_bool: true,
            column_float: 1.0,
            column_complex: vec![vec![vec![[1, index as u8, 3]]]],
        };

        let mut writer = SqliteDatasetWriter::<Complex>::new(&db_file, true).unwrap();
        for index in 0..1000 {
            writer.write("train", &item(index)).unwrap();
        }
        writer.set_completed().unwrap();

        let dataset = SqliteDataset::<Complex>::from_db_file(&db_file, "train").unwrap();

        let requested: Vec<usize> = (0..8000).map(|i| i % 1000).collect();
        let got: Vec<i64> = requested
            .par_iter()
            .map(|&index| dataset.get(index).unwrap().column_int)
            .collect();
        for (position, &index) in requested.iter().enumerate() {
            assert_eq!(got[position], index as i64);
        }

        // `get_many` takes its own connection out of the pool, so exercise it concurrently too.
        (0..256usize).into_par_iter().for_each(|batch| {
            let indexes: Vec<usize> = (0..32).map(|offset| (batch * 32 + offset) % 1000).collect();
            let items = dataset.get_many(indexes.clone()).unwrap();
            assert_eq!(items.len(), indexes.len());
            for (item, index) in items.iter().zip(&indexes) {
                assert_eq!(item.column_int, *index as i64);
            }
        });
    }

    /// A `set_completed` that fails must not let a retry publish an unfinished database.
    ///
    /// The checkpoint refuses to run while another connection holds a read transaction open. That
    /// refusal is what stops a half-written dataset from being published, so the retry after it
    /// has to refuse too rather than fall through to the rename.
    #[rstest]
    pub fn sqlite_writer_set_completed_is_retryable(writer_fixture: (Writer, TempDir)) {
        let (writer, _tmp_dir) = writer_fixture;
        let item = Complex {
            column_str: "HI".to_string(),
            column_bytes: vec![1, 2, 3],
            column_int: 0,
            column_bool: true,
            column_float: 1.0,
            column_complex: vec![vec![vec![[1, 2, 3]]]],
        };
        for _ in 0..500 {
            writer.write("train", &item).unwrap();
        }

        let mut writer = writer;

        let blocker = open_database(tmp_db_file(&writer.db_file), true)
            .unwrap()
            .connect()
            .unwrap();
        block_on(async {
            blocker.execute("BEGIN", ()).await?;
            blocker
                .prepare("select count(*) from train")
                .await?
                .query_row(())
                .await
        })
        .unwrap();

        let first = writer.set_completed();
        assert!(first.is_err(), "checkpoint should have been refused");
        assert!(!writer.db_file.exists(), "nothing may be published yet");

        // The retry must refuse as well, rather than publishing a truncated database.
        assert!(
            writer.set_completed().is_err(),
            "retry published while the checkpoint was still refused"
        );
        assert!(!writer.db_file.exists(), "nothing may be published yet");

        // The writer is still usable, and still reports errors rather than panicking.
        writer.write("train", &item).unwrap();

        // Once nothing holds the database, completing succeeds with every row intact.
        drop(blocker);
        writer
            .set_completed()
            .expect("should complete once the database is no longer busy");

        let dataset = SqliteDataset::<Complex>::from_db_file(&writer.db_file, "train").unwrap();
        assert_eq!(dataset.len(), 501);
        assert!(writer.write("train", &item).is_err());
    }

    /// Writes enough rows to span several transactions, then reads every one of them back.
    ///
    /// Rows live in the write-ahead log until `set_completed` checkpoints them into the database
    /// file, and only a run longer than [`WRITE_BATCH_SIZE`] exercises both the batch boundary and
    /// a checkpoint with more than one batch to fold in.
    #[rstest]
    pub fn sqlite_writer_write_across_batches(writer_fixture: (Writer, TempDir)) {
        let (writer, _tmp_dir) = writer_fixture;
        let record_count = WRITE_BATCH_SIZE * 2 + 5;

        let item = |index: usize| Complex {
            column_str: format!("item_{index}"),
            column_bytes: vec![index as u8, 2, 3],
            column_int: index as i64,
            column_bool: true,
            column_float: 1.0,
            column_complex: vec![vec![vec![[1, index as u8, 3]]]],
        };

        for index in 0..record_count {
            assert_eq!(writer.write("train", &item(index)).unwrap(), index);
        }

        let mut writer = writer;
        writer.set_completed().expect("Failed to set completed");

        let train = SqliteDataset::<Complex>::from_db_file(&writer.db_file, "train").unwrap();
        assert_eq!(train.len(), record_count);
        assert_eq!(train.get(0).unwrap(), item(0));
        assert_eq!(train.get(WRITE_BATCH_SIZE).unwrap(), item(WRITE_BATCH_SIZE));
        assert_eq!(train.get(record_count - 1).unwrap(), item(record_count - 1));

        // The write-ahead log must not outlive the completed dataset.
        assert!(!wal_file(&tmp_db_file(&writer.db_file)).exists());
    }
}
