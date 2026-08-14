"""Tests for the NSHM database comparator."""

import sqlite3
from pathlib import Path

import pytest
import typer

from workflow.scripts import compare_nshmdb as cn


def build_db(path: Path, rows: list[tuple[int, str]]) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE rupture (i INTEGER, s TEXT)")
    connection.executemany("INSERT INTO rupture VALUES (?, ?)", rows)
    connection.commit()
    return connection


def test_table_digest_is_independent_of_row_order(tmp_path: Path) -> None:
    left = build_db(tmp_path / "left.db", [(1, "x"), (2, "y"), (3, "z")])
    right = build_db(tmp_path / "right.db", [(3, "z"), (1, "x"), (2, "y")])

    assert cn.table_digest(left, "rupture") == cn.table_digest(right, "rupture")


def test_table_digest_detects_a_changed_row(tmp_path: Path) -> None:
    left = build_db(tmp_path / "left.db", [(1, "x"), (2, "y")])
    right = build_db(tmp_path / "right.db", [(1, "x"), (2, "CHANGED")])

    assert cn.table_digest(left, "rupture") != cn.table_digest(right, "rupture")


def test_table_digest_detects_a_duplicated_row(tmp_path: Path) -> None:
    left = build_db(tmp_path / "left.db", [(1, "x"), (2, "y")])
    right = build_db(tmp_path / "right.db", [(1, "x"), (2, "y"), (2, "y")])

    assert cn.table_digest(left, "rupture") != cn.table_digest(right, "rupture")


def test_table_digest_is_not_blind_to_even_multiplicity(tmp_path: Path) -> None:
    # XOR cancels a row against its own duplicate, so both of these would digest
    # to zero and compare equal. The row counts match, so the digest is the only
    # thing that can separate them -- which is what makes this, and not the test
    # above, the one that actually pins the summation.
    left = build_db(tmp_path / "left.db", [(1, "x"), (1, "x")])
    right = build_db(tmp_path / "right.db", [(2, "y"), (2, "y")])

    assert cn.table_digest(left, "rupture") != cn.table_digest(right, "rupture")


def test_table_names_are_sorted(tmp_path: Path) -> None:
    connection = build_db(tmp_path / "db.db", [])
    connection.execute("CREATE TABLE fault (i INTEGER)")
    connection.commit()

    assert cn.table_names(connection) == ["fault", "rupture"]


def test_main_exits_0_on_logically_identical_databases(tmp_path: Path) -> None:
    # Different insertion order, so the files are not byte-identical -- which is
    # the whole point: SQLite page layout need not be stable across builds.
    build_db(tmp_path / "left.db", [(1, "x"), (2, "y"), (3, "z")])
    build_db(tmp_path / "right.db", [(3, "z"), (2, "y"), (1, "x")])

    cn.compare_nshmdb(tmp_path / "left.db", tmp_path / "right.db")


def test_main_exits_1_when_a_table_differs(tmp_path: Path) -> None:
    build_db(tmp_path / "left.db", [(1, "x"), (2, "y")])
    build_db(tmp_path / "right.db", [(1, "x"), (2, "CHANGED")])

    with pytest.raises(typer.Exit) as exit_info:
        cn.compare_nshmdb(tmp_path / "left.db", tmp_path / "right.db")

    assert exit_info.value.exit_code == 1


def test_main_exits_1_when_the_table_sets_differ(tmp_path: Path) -> None:
    build_db(tmp_path / "left.db", [(1, "x")])
    right = build_db(tmp_path / "right.db", [(1, "x")])
    right.execute("CREATE TABLE fault (i INTEGER)")
    right.commit()

    with pytest.raises(typer.Exit) as exit_info:
        cn.compare_nshmdb(tmp_path / "left.db", tmp_path / "right.db")

    assert exit_info.value.exit_code == 1
