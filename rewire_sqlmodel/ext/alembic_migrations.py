import base64
from contextlib import contextmanager
import os
from pathlib import Path
import random
from shutil import copytree
from typing import Any, Iterable
from uuid import uuid4
import alembic.command
from loguru import logger
from sqlalchemy import (
    Column,
    Constraint,
    ForeignKeyConstraint,
    Index,
    MetaData,
    PrimaryKeyConstraint,
    Table,
)
from sqlmodel import SQLModel
from rewire.config import ConfigDependency
from rewire.context import CTX
from rewire.plugins import simple_plugin
from alembic.config import Config as AlembicConfig
from sqlalchemy.ext.asyncio import AsyncEngine
from rewire.store import SimpleStore
from rewire_sqlmodel import test_connection
from alembic.script import ScriptDirectory

plugin = simple_plugin(bind=False, load=False)

ALEMBIC_LOCATION = Path("alembic")


@plugin.bind
class Config(ConfigDependency):
    __location__ = "...alembic"
    __fallback__ = {}

    generate: bool = False
    migration_type_ignore: bool = True
    schema_migrations: bool = False


class Formatter:
    ctx = CTX()

    value: int = 0
    imports: dict[str, str]

    def __init__(self) -> None:
        self.imports = {}

    @classmethod
    def use(cls):
        return cls.ctx.use(cls.ctx.get(Formatter()))

    @classmethod
    def wrap(
        cls, data: Any, end: str = "", newline: bool = True, start: str = ""
    ) -> str:
        self = cls.ctx.get()
        return self.value * "    " + start + str(data) + end + "\n" * newline

    @classmethod
    def repr(
        cls, data: Any, end: str = "", newline: bool = True, start: str = ""
    ) -> str:
        self = cls.ctx.get()
        self.add_import(data)
        return self.value * "    " + start + repr(data) + end + "\n" * newline

    @classmethod
    def add_import(cls, obj: Any):
        self = cls.ctx.get()
        t = type(obj)
        if t.__module__ != "builtins":
            if self.imports.get(t.__name__, t.__module__) != t.__module__:
                raise RuntimeError(
                    f"Name conflict with {t.__module__} and {self.imports[t.__name__]}"
                )
            self.imports[t.__name__] = t.__module__
        return t.__name__

    @classmethod
    @contextmanager
    def ident(cls, value: int | None = 1):
        with cls.use() as self:
            prev_value = self.value
            if value is None:
                self.value = 0
            else:
                self.value += value
            yield self
            self.value = prev_value

    def format_imports(self):
        imports = ""
        for obj, source in self.imports.items():
            imports += f"from {source} import {obj}\n"
        return imports


def _dump_column_list(columns: Iterable[Column | str]):
    out = "["
    for column in columns:
        if isinstance(column, Column):
            column = column.name
        out += repr(column)
    out += "]"
    return out


def format_dialect(obj):
    # if you are going to support it also support in get_{}_migration
    assert not obj.dialect_kwargs, "Dialect kwargs are not supported"
    assert not obj.dialect_options, "Dialect options are not supported"
    return ""


def format_fk_name(fk: ForeignKeyConstraint):
    assert "fk" in SQLModel.metadata.naming_convention
    convention = SQLModel.metadata.naming_convention["fk"]
    assert isinstance(convention, str)

    name = convention % dict(
        table_name=fk.columns[0].table,
        column_0_name=fk.columns[0].name,
        referred_table_name=fk.elements[0].column.table,
    )
    return fk.name or name


def dump_pk_constraint(pk: PrimaryKeyConstraint, end: str = ""):
    out = Formatter.wrap(f"{Formatter.add_import(pk)}(")
    with Formatter.ident():
        for column in pk.columns:
            if isinstance(column, Column):
                column = column.name
            out += Formatter.repr(column, ",")
        if pk.name:
            out += Formatter.wrap(f"name={pk.name!r},")
        out += format_dialect(pk)
    out += Formatter.wrap(")", end)
    return out


def dump_fk_constraint(pk: ForeignKeyConstraint, end: str = ""):
    out = Formatter.wrap(f"{Formatter.add_import(pk)}(")
    with Formatter.ident():
        out += Formatter.wrap(f"{_dump_column_list(pk.columns)}", ",")
        out += dump_statement(
            [f"{x.column.table.name}.{x.column.name}" for x in pk.elements], ","
        )

        out += Formatter.wrap(f"name={format_fk_name(pk)!r},")
        out += format_dialect(pk)
    out += Formatter.wrap(")", end)
    return out


def dump_index(index: Index, end=""):
    out = Formatter.wrap(f"{Formatter.add_import(index)}(")
    with Formatter.ident():
        out += Formatter.wrap(f"{index.name!r},")
        for column in index.columns:
            out += Formatter.repr(column.name, ",")
        out += Formatter.wrap(f"unique={index.unique!r},")
        out += format_dialect(index)
    out += Formatter.wrap(")", end)
    return out


def dump_table(table: Table, end: str = "", include_meta: bool = True):
    out = Formatter.wrap(f"{Formatter.add_import(table)}(")
    with Formatter.ident():
        out += Formatter.repr(table.name, ",")
        if include_meta:
            out += Formatter.repr(MetaData(), ",")
        for column in table.columns:
            out += dump_statement(column, ",")
        for constraint in table.constraints:
            out += dump_statement(constraint, ",")
        for index in table.indexes:
            out += dump_statement(index, ",")
        out += format_dialect(table)
    out += Formatter.wrap(")", end)
    return out


def dump_column(column: Column, end: str = ""):
    out = Formatter.wrap(f"{Formatter.add_import(column)}(")

    with Formatter.ident():
        if column.comment:
            out = Formatter.wrap(f"# {column.comment}")
        out += Formatter.repr(column.name, ",")
        out += dump_statement(column.type, ",")
        if column.key != column.name:
            out += Formatter.wrap(f"key={column.key},")
        if column.primary_key:
            out += Formatter.wrap(f"primary_key={column.primary_key},")
        out += Formatter.wrap(f"nullable={column.nullable},")
        if column.server_default:
            logger.warning("Migrations with server_default may not work correctly")
            out += Formatter.wrap(f"server_default={column.server_default!r},")
        out += format_dialect(column)

    out += Formatter.wrap(")", end)
    return out


def dump_list(data: list, end: str = ""):
    if not data:
        return Formatter.wrap("[]", end)
    out = Formatter.wrap("[")
    with Formatter.ident():
        for item in data:
            out += dump_statement(item, end=",")
    out += Formatter.wrap("]", end)
    return out


def dump_dict(data: dict, end: str = ""):
    if not data:
        return Formatter.wrap("{}", end)
    out = Formatter.wrap("{")
    with Formatter.ident():
        for key, value in data.items():
            out += Formatter.repr(key, newline=False)
            out += ": "
            out += dump_statement(value, end=",").strip(" ")
    out += Formatter.wrap("}", end)
    return out


def dump_statement(
    obj: Any,
    end: str = "",
    with_imports: bool = False,
    start: str = "",
    also_import: dict[str, str] = {},
) -> str:
    with Formatter.use() as formatter:
        match obj:
            case Table():
                out = dump_table(obj, end)
            case PrimaryKeyConstraint():
                out = dump_pk_constraint(obj, end)
            case Index():
                out = dump_index(obj, end)
            case ForeignKeyConstraint():
                out = dump_fk_constraint(obj, end)
            case Column():
                out = dump_column(obj, end)
            case list():
                out = dump_list(obj, end)
            case dict():
                out = dump_dict(obj, end)
            case _:
                out = Formatter.repr(obj, end)
        if start:
            out = start + out
        if with_imports:
            formatter.imports.update(also_import)
            out = f"{formatter.format_imports()}\n\n{out}"
    return out


def dump_schema():
    tables: dict[str, Table] = {}
    for table in SQLModel.metadata.sorted_tables:
        tables[table.name] = table
    return (
        dump_statement(
            tables,
            with_imports=True,
            start="_Meta = MetaData()\nschema = ",
            also_import={"MetaData": "sqlalchemy.sql.schema"},
        )
        .replace("MetaData()", "_Meta")
        .replace("= _Meta", "= MetaData()", 1)
    )


@plugin.setup(stage=-10, dependencies=[test_connection])
def detect_alembic(_: AsyncEngine):
    global alembicExists
    try:
        open(ALEMBIC_LOCATION / "env.py").close()
        return True
    except FileNotFoundError:
        return False


@plugin.setup(stage=-10, dependencies=[detect_alembic])
def setup_alembic_config() -> AlembicConfig:
    return AlembicConfig(ALEMBIC_LOCATION / "config.ini")


@plugin.setup(stage=-10)
def pre_upgrade_db(cfg: AlembicConfig, exists: detect_alembic.Result):
    if not exists:
        logger.warning(
            "Skipping alembic pre-migrate because alembic is not initialized"
        )
        return
    alembic.command.upgrade(cfg, revision="head")


@plugin.setup(stage=-10, dependencies=[pre_upgrade_db])
def init_alembic(
    cgf: AlembicConfig, alembic_exists: detect_alembic.Result, config: Config.Value
):
    if not alembic_exists and config.generate:
        alembic.command.init(cgf, directory="alembic", template="async")
        templates = Path(__file__).parent / "alembic_template"
        copytree(templates, "alembic", dirs_exist_ok=True)


@plugin.setup(stage=-10, dependencies=[init_alembic])
def generate_migrations(cfg: AlembicConfig, config: Config.Value):
    prefix = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    rev = prefix + base64.urlsafe_b64encode(uuid4().bytes).rstrip(b"=").decode(
        "ascii"
    ).replace("-", "")
    script = ScriptDirectory.from_config(cfg)
    head_rev = script.get_current_head()
    assert head_rev
    head = script.get_revision(head_rev).module
    with Formatter.use() as schema_fmt:
        schema_imports, schema = dump_schema().split("\n\n", 1)

    if (
        not hasattr(head, "schema") and config.schema_migrations
    ) or not config.schema_migrations:
        alembic.command.revision(cfg, autogenerate=True, message="Dev", rev_id=rev)
        filename = ALEMBIC_LOCATION / "versions" / f"{rev}_dev.py"
        with open(filename) as f:
            data = f.read()
        if config.migration_type_ignore:
            with open(filename, "w") as f:
                data = f"# type: ignore\n{data}"
                f.write(data)
                f.write("\n")
                if config.schema_migrations:
                    logger.warning("Migrating to schema based migrations")
                    f.write(schema_imports + "\n" + schema)
        if data.count("pass") == 2 and not config.schema_migrations:
            os.remove(filename)
        return

    __store__ = SimpleStore()
    exec(
        f"{schema_imports}\n{schema}\n\n__store__.set(schema)", {"__store__": __store__}
    )
    current_schema: dict[str, Table] = __store__.get()
    prev_schema: dict[str, Table] = head.schema

    with Formatter.ident() as fmt:
        fmt.imports.update(schema_fmt.imports)
        up_cmds, dn_cmds = get_schema_migrations(current_schema, prev_schema)

    out = "from alembic import op\n"
    out += fmt.format_imports()
    out += "\n\n"

    out += "# revision identifiers, used by Alembic.\n"
    out += f"revision = {rev!r}\n"
    out += f"down_revision = {head_rev!r}\n"
    out += "branch_labels = None\n"
    out += "depends_on = None\n\n\n"

    out += "def upgrade() -> None:\n"
    with Formatter.ident():
        out += Formatter.wrap(
            "# ### commands auto generated by rewire_sqlmodel - please adjust! ###"
        )
        out += up_cmds
        out += Formatter.wrap("# ### end Alembic commands ###")
    out += "\n\ndef downgrade() -> None:\n"
    with Formatter.ident():
        out += Formatter.wrap(
            "# ### commands auto generated by rewire_sqlmodel - please adjust! ###"
        )
        out += dn_cmds
        out += Formatter.wrap("# ### end Alembic commands ###\n\n")

    out += schema
    if dn_cmds:
        (ALEMBIC_LOCATION / "versions" / f"{rev}_dev.py").write_text(out)


def get_schema_migrations(
    current_schema: dict[str, Table], prev_schema: dict[str, Table]
):
    tables = set(current_schema) | set(prev_schema)
    up_cmds = ""
    dn_cmds = ""

    for table in sorted(tables):
        if table in current_schema and table in prev_schema:  # table update/noop
            prev_table = prev_schema[table]
            curr_table = current_schema[table]
            with Formatter.ident():
                up_cmds_, dn_cmds_ = get_table_migration(prev_table, curr_table)
            if up_cmds_:
                up_cmds += Formatter.wrap(
                    f"with op.batch_alter_table({table!r}, schema=None) as batch_op:"
                )
                up_cmds += up_cmds_
                dn_cmds += Formatter.wrap(
                    f"with op.batch_alter_table({table!r}, schema=None) as batch_op:"
                )
                dn_cmds += dn_cmds_
        elif table in current_schema:  # table create
            up_cmds += dump_table(current_schema[table], include_meta=False).replace(
                "Table", "op.create_table", 1
            )
            dn_cmds += Formatter.wrap(f"op.drop_table({table!r})")
        else:  # table delete
            dn_cmds += dump_table(prev_schema[table], include_meta=False).replace(
                "Table", "op.create_table", 1
            )
            up_cmds += Formatter.wrap(f"op.drop_table({table!r})")
    return up_cmds, dn_cmds


def get_table_migration(prev_table: Table, curr_table: Table):
    up_cmds = ""
    dn_cmds = ""

    prev_indexes: dict[str | None, Index] = {x.name: x for x in prev_table.indexes}
    curr_indexes: dict[str | None, Index] = {x.name: x for x in curr_table.indexes}
    assert (
        None not in curr_indexes
    ), f"Unnamed indexes are not supported, {curr_indexes[None]}"
    indexes = set(prev_indexes) | set(curr_indexes)
    for index in indexes:
        up_cmds_, dn_cmds_ = get_index_migration(prev_indexes, curr_indexes, index)
        up_cmds += up_cmds_
        dn_cmds += dn_cmds_

    prev_constraints: dict[str, Constraint] = {
        dump_statement(x): x for x in prev_table.constraints
    }
    curr_constraints: dict[str, Constraint] = {
        dump_statement(x): x for x in curr_table.constraints
    }
    constraints = set(prev_constraints) | set(curr_constraints)
    for constraint in constraints:
        up_cmds_, dn_cmds_ = get_constraint_migration(
            prev_constraints, curr_constraints, constraint
        )
        up_cmds += up_cmds_
        dn_cmds += dn_cmds_

    prev_columns: dict[str, Column] = {x.name: x for x in prev_table.columns}
    curr_columns: dict[str, Column] = {x.name: x for x in curr_table.columns}
    columns = set(prev_columns) | set(curr_columns)
    for column in columns:
        up_cmds_, dn_cmds_ = get_column_migration(prev_columns, curr_columns, column)
        up_cmds += up_cmds_
        dn_cmds += dn_cmds_

    return up_cmds, dn_cmds


def get_constraint_migration(
    prev_constraints: dict[str, Constraint],
    curr_constraints: dict[str, Constraint],
    constraint: str,
):
    up_cmds = dn_cmds = ""

    def _create_constr(constraint: Constraint):
        match constraint:
            case ForeignKeyConstraint():
                out = Formatter.wrap("batch_op.create_foreign_key(")
                with Formatter.ident():
                    out += Formatter.repr(
                        format_fk_name(constraint), end=",", start="constraint_name="
                    )
                    out += Formatter.repr(
                        constraint.referred_table.name, end=",", start="referent_table="
                    )
                    out += Formatter.wrap(
                        _dump_column_list(constraint.columns),
                        end=",",
                        start="local_cols=",
                    )
                    out += Formatter.repr(
                        [
                            f"{x.column.table.name}.{x.column.name}"
                            for x in constraint.elements
                        ],
                        end=",",
                        start="remote_cols=",
                    )
                out += Formatter.wrap(")")
                return out
            case PrimaryKeyConstraint():
                raise NotImplementedError("Unable to upgrade pk")

        raise RuntimeError(f"Unable to create {constraint}")

    def _delete_constr(constraint: Constraint):
        match constraint:
            case ForeignKeyConstraint():
                return Formatter.wrap(
                    f'batch_op.drop_constraint(constraint_name={format_fk_name(constraint)!r}, type_="foreignkey")'
                )
            case PrimaryKeyConstraint():
                return ""

        raise RuntimeError(f"Unable to delete {constraint}")

    if constraint in prev_constraints and constraint in curr_constraints:
        assert dump_statement(prev_constraints[constraint]) == dump_statement(
            curr_constraints[constraint]
        ), f"Unable to update {curr_constraints[constraint]=!r}"
    elif constraint in curr_constraints:
        up_cmds += _create_constr(curr_constraints[constraint])
        dn_cmds += _delete_constr(curr_constraints[constraint])
    else:
        up_cmds += _delete_constr(prev_constraints[constraint])
        dn_cmds += _create_constr(prev_constraints[constraint])
    return up_cmds, dn_cmds


def get_index_migration(
    prev_indexes: dict[str | None, Index],
    curr_indexes: dict[str | None, Index],
    index: str | None,
):
    up_cmds = dn_cmds = ""
    if index in prev_indexes and index in curr_indexes:
        assert dump_statement(prev_indexes[index]) == dump_statement(
            curr_indexes[index]
        ), f"Unable to update {curr_indexes[index]=!r}"
    elif index in curr_indexes:
        idx = curr_indexes[index]

        up_cmds += Formatter.wrap("batch_op.create_index(")
        with Formatter.ident():
            up_cmds += Formatter.repr(idx.name, end=",")
            up_cmds += Formatter.wrap(_dump_column_list(idx.columns), end=",")
            up_cmds += Formatter.repr(idx.unique, start="unique=", end=",")
        up_cmds += Formatter.wrap(")")
        dn_cmds += Formatter.wrap(f"batch_op.drop_index({idx.name!r})")
    else:
        idx = prev_indexes[index]

        up_cmds += Formatter.wrap(f"batch_op.drop_index({idx.name!r})")
        dn_cmds += Formatter.wrap("batch_op.create_index(")
        with Formatter.ident():
            dn_cmds += Formatter.repr(idx.name, end=",")
            dn_cmds += Formatter.wrap(_dump_column_list(idx.columns), end=",")
            dn_cmds += Formatter.repr(idx.unique, start="unique=", end=",")
        dn_cmds += Formatter.wrap(")")
    return up_cmds, dn_cmds


def get_column_migration(
    prev_columns: dict[str, Column],
    curr_columns: dict[str, Column],
    column: str,
):
    up_cmds = dn_cmds = ""
    if column in curr_columns and column in prev_columns:  # column update/noop
        prev_column = prev_columns[column]
        curr_column = curr_columns[column]
        if type(prev_column) != type(curr_column) or dump_statement(
            prev_column
        ) != dump_statement(curr_column):
            up_cmds += Formatter.wrap("batch_op.alter_column(")
            dn_cmds += Formatter.wrap("batch_op.alter_column(")
            with Formatter.ident():
                up_cmds += Formatter.repr(column, end=",")
                up_cmds += Formatter.repr(curr_column.type, end=",", start="type_=")
                up_cmds += Formatter.repr(
                    curr_column.nullable, end=",", start="nullable="
                )
                dn_cmds += Formatter.repr(column, end=",")
                dn_cmds += Formatter.repr(prev_column.type, end=",", start="type_=")
                dn_cmds += Formatter.repr(
                    prev_column.nullable, end=",", start="nullable="
                )
                # sorry no MySQL support :(
            up_cmds += Formatter.wrap(")")
            dn_cmds += Formatter.wrap(")")
    elif column in curr_columns:  # column create
        up_cmds += Formatter.wrap("batch_op.add_column(")
        with Formatter.ident():
            up_cmds += dump_column(curr_columns[column], end=",")
        up_cmds += Formatter.wrap(")")
        dn_cmds += Formatter.wrap(f"batch_op.drop_column({column!r})")
    else:  # column delete
        up_cmds += Formatter.wrap(f"batch_op.drop_column({column!r})")
        dn_cmds += Formatter.wrap("batch_op.add_column(")
        with Formatter.ident():
            dn_cmds += dump_column(prev_columns[column], end=",")
        dn_cmds += Formatter.wrap(")")
    return up_cmds, dn_cmds


@plugin.setup(stage=-10, dependencies=[generate_migrations])
def upgrade_db(
    cfg: AlembicConfig, alembic_exists: detect_alembic.Result, config: Config.Value
):
    if not config.generate:
        return
    if alembic_exists:
        alembic.command.upgrade(cfg, revision="head")
