#[macro_export]
/// Display information with special formatting to indicate an info message.
macro_rules! info {
    ($($t:tt)*) => {{
        eprint!("\x1b[90m[RUSTINFO] ");
        eprint!($($t)*);
        eprintln!("\x1b[0m");
    }};
}