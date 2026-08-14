#!/usr/bin/env python3
"""Compare two NSHM databases on logical content.

Description
-----------
Byte-identity is not expected and is not the test: SQLite page layout and rowid
allocation need not be stable across builds of the same data. What must match is
the content. Each table's rows are hashed individually and the digests summed,
which is independent of the order SQLite returns them in, sensitive to a row's
multiplicity (an XOR would not be), and uses constant memory -- which matters,
because ``rupture_faults`` has around 20 million rows.

Usage
-----
``compare-nshmdb LEFT_DB RIGHT_DB``
"""

import hashlib
import sqlite3
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


def table_names(connection: sqlite3.Connection) -> list[str]:
    """Return the names of the tables in a database, sorted.

    Parameters
    ----------
    connection : sqlite3.Connection
        An open connection to the database.

    Returns
    -------
    list of str
        The sorted table names.
    """
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    )
    return [name for (name,) in rows]


def table_digest(connection: sqlite3.Connection, table: str) -> tuple[int, str]:
    """Return the row count and an order-independent content hash for one table.

    Parameters
    ----------
    connection : sqlite3.Connection
        An open connection to the database.
    table : str
        Name of the table to digest. Must come from :func:`table_names`.

    Returns
    -------
    tuple of (int, str)
        The number of rows, and a 64-character hex content digest.
    """
    total = 0
    count = 0
    for row in connection.execute(f'SELECT * FROM "{table}"'):
        row_digest = hashlib.sha256(repr(row).encode("utf-8")).digest()
        total = (total + int.from_bytes(row_digest, "big")) % (2**256)
        count += 1
    return count, f"{total:064x}"


@app.command()
def compare_nshmdb(
    left: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    right: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Compare two NSHM databases on logical content, exiting 1 if they differ.

    Parameters
    ----------
    left : Path
        The first database.
    right : Path
        The second database.
    """
    left_connection = sqlite3.connect(left)
    right_connection = sqlite3.connect(right)

    left_tables = table_names(left_connection)
    right_tables = table_names(right_connection)

    differences: list[str] = []
    if left_tables != right_tables:
        differences.append(f"table sets differ: {left_tables} vs {right_tables}")

    for table in sorted(set(left_tables) & set(right_tables)):
        left_count, left_hash = table_digest(left_connection, table)
        right_count, right_hash = table_digest(right_connection, table)
        same = (left_count, left_hash) == (right_count, right_hash)
        print(
            f"{table:36s} {left_count:>11,} rows  {left_hash[:16]}  "
            f"{'same' if same else 'DIFFER'}"
        )
        if not same:
            differences.append(
                f"{table}: {left_count:,} rows / {left_hash[:16]} vs "
                f"{right_count:,} rows / {right_hash[:16]}"
            )

    if differences:
        print("\nDATABASES DIFFER:")
        for difference in differences:
            print(f"  - {difference}")
        raise typer.Exit(code=1)

    print("\nDatabases are logically identical.")


if __name__ == "__main__":
    app()
