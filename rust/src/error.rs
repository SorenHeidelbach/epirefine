//! One error type for the kernels; converted to Python exceptions in lib.rs.
use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// read failures, truncated or corrupt input -> OSError
    Io(String),
    /// malformed content or arguments -> ValueError
    Value(String),
    /// a count does not fit the output type -> OverflowError
    Overflow(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(s) | Error::Value(s) | Error::Overflow(s) => f.write_str(s),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

pub fn value<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Value(msg.into()))
}

pub fn io<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Io(msg.into()))
}
