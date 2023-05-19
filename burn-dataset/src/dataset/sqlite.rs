use std::marker::PhantomData;

use crate::Dataset;

use r2d2::Pool;
use r2d2_sqlite::{rusqlite::OpenFlags, SqliteConnectionManager};
use serde::de::DeserializeOwned;
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
    phantom: PhantomData<I>,
}

impl<I> SqliteDataset<I> {
    pub fn from_db_file(db_file: &str, split: &str) -> Self {
        // Create a connection pool
        let conn_pool = create_conn_pool(db_file);

        // Create a select statement and save it
        let select_statement = format!("select * from {split} where row_id = ?");

        // Save the column names and the number of rows
        let (columns, len) = fetch_columns_and_len(&conn_pool, &select_statement, split);

        SqliteDataset {
            db_file: db_file.to_string(),
            split: split.to_string(),
            conn_pool,
            columns,
            len,
            select_statement,
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

        // Query the row with the given row_id and deserialize it into I using column names (fast option)
        let mut rows = statement
            .query_and_then([row_id], |row| {
                from_row_with_columns::<I>(row, &self.columns)
            })
            .unwrap();

        // Return the first row if found else None (error)
        rows.next().and_then(|res| match res {
            Ok(val) => Some(val),
            Err(_) => None,
        })
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
fn create_conn_pool(db_file: &str) -> Pool<SqliteConnectionManager> {
    // Create a connection pool and make sure the connections are read only
    let manager =
        SqliteConnectionManager::file(db_file).with_flags(OpenFlags::SQLITE_OPEN_READ_ONLY);
    let conn_pool: Pool<SqliteConnectionManager> = Pool::new(manager).unwrap();
    conn_pool
}

#[cfg(test)]
mod tests {
    use rayon::prelude::*;
    use rstest::{fixture, rstest};
    use serde::{Deserialize, Serialize};

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
        SqliteDataset::from_db_file("tests/data/sqlite-dataset.db", "train")
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
}
