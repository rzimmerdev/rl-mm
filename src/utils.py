from typing import Dict

from rich.console import Group
from rich.live import Live
from rich.table import Table


class SimpleTable:
    def __init__(self, title, columns=None):
        self.title = title
        self.columns = columns or []
        self.rows = []

    def add_row(self, *args):
        self.rows.append(args)

    def add_column(self, column_name, width=20):
        self.columns.append((column_name, width))

    def clear(self):
        self.rows = []

    def update(self):
        table = Table(title=self.title)
        for column in self.columns:
            table.add_column(*column)

        for row in self.rows:
            table.add_row(*row)

        return table


class Visualizer:
    def __init__(self, refresh_per_second=4):
        self.tables: Dict[str, SimpleTable] = {}
        self.live = None
        self.refresh_per_second = refresh_per_second

    def add_column(self, table_id, column_name, width=20):
        if table_id not in self.tables:
            raise KeyError(f"Table with ID '{table_id}' does not exist.")
        table = self.tables[table_id]
        table.add_column(column_name, width=width)

    def add_table(self, table_id, title, columns = None):
        if table_id in self.tables:
            raise KeyError(f"Table with ID '{table_id}' already exists.")
        self.tables[table_id] = SimpleTable(title, columns)

    def update(self):
        tables_group = Group(*self.tables.values())
        self.live.update(tables_group)

    def __enter__(self):
        self.live = Live(refresh_per_second=self.refresh_per_second)
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.live.__exit__(exc_type, exc_val, exc_tb)
