from contextlib import asynccontextmanager
from functools import wraps
from types import NoneType, UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Awaitable,
    Callable,
    Literal,
    Self,
    Type,
    get_args,
    get_origin,
)
import typing
from uuid import UUID
import anyio
from loguru import logger

from pydantic import BaseModel, TypeAdapter
from pydantic.config import ConfigDict
from sqlalchemy import JSON, BigInteger, TypeDecorator
from sqlalchemy.engine.interfaces import IsolationLevel, _ParamStyle
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import _ResetStyleArgType
from sqlmodel import Field, SQLModel as BaseSQLModel, col
from sqlmodel import select
from typing_extensions import Unpack

from rewire.config import ConfigDependency, config
from rewire.context import CTX, Context, use_context_value
from rewire.dependencies import Dependencies, TypeRef
from rewire.lifecycle import LifecycleModule
from rewire.plugins import Plugin, simple_plugin
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel.sql.expression import SelectOfScalar
from sqlalchemy.engine.result import ScalarResult
from sqlalchemy.orm import Mapped

plugin = simple_plugin()
_Debug = Literal["debug"]


@plugin.bind
class ConnectionConfig(ConfigDependency):
    url: str
    connect_args: dict[Any, Any] = {}
    echo: bool | Any = False
    echo_pool: bool | _Debug = False
    enable_from_linting: bool = True
    encoding: str = "utf-8"
    execution_options: dict[Any, Any] = {}
    future: bool = True
    hide_parameters: bool = False
    implicit_returning: bool = True
    isolation_level: IsolationLevel | None = None
    label_length: int | None = None
    logging_name: str | None = None
    max_identifier_length: int | None = None
    max_overflow: int = 10
    module: Any | None = None
    paramstyle: _ParamStyle | None = None
    pool_logging_name: str | None = None
    pool_pre_ping: bool = False
    pool_size: int = 5
    pool_recycle: int = -1
    pool_reset_on_return: _ResetStyleArgType = "rollback"
    pool_timeout: float = 30
    pool_use_lifo: bool = False
    plugins: list[str] | None = None
    query_cache_size: int | None = None
    kwargs: dict[str, Any] = {}

    expire_on_commit: bool = True

    @property
    def allow_multithread(self):
        return not self.url.startswith("sqlite")


@config(fallback={})
class PluginConfig(BaseModel):
    patch_types: bool = True

    parallel: bool | None = None


@plugin.bind
class ConventionConfig(ConfigDependency):
    convention: dict[str, str] = {
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
    patchConventions: bool = True


@plugin.setup(stage=-50)
async def patch_conventions(config: ConventionConfig.Value):
    if config.patchConventions:
        BaseSQLModel.metadata.naming_convention = (
            dict(BaseSQLModel.metadata.naming_convention) | config.convention
        )


class MigrationConfig(BaseModel):
    disable: bool = False
    alembic: dict | None = None


class SQLModelConfigDict(ConfigDict, total=False):
    table: bool


class SQLModel(BaseSQLModel):
    if TYPE_CHECKING:

        def __init_subclass__(cls, **kwargs: Unpack[SQLModelConfigDict]):
            return super().__init_subclass__(**kwargs)

    def add(self):
        session_context.get().add(self)
        return self

    async def delete(self):
        await session_context.get().delete(self)
        return self

    @classmethod
    def select(cls) -> "SelectOfScalarExtended[Self]":
        return SelectOfScalarExtended(cls)


class ModelMapped[T](Mapped[T]):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        item_tp = get_args(source_type)[0]
        return handler.generate_schema(item_tp)


BigInt = int

if PluginConfig.patch_types:
    from pydantic.fields import FieldInfo
    from sqlalchemy.util import memoized_property
    from sqlalchemy_utils import UUIDType
    from sqlmodel.main import get_sqlalchemy_type as _get_sqlalchemy_type

    dialect_patched = set()

    class PydanticJSON(TypeDecorator):
        impl = JSON

        cache_ok = True

        def __init__(self, type: Type = None, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._type = type

        def bind_processor(self, dialect):
            json = JSON()

            def test(a):
                return self.type_adapter.dump_json(a).decode()

            return json._make_bind_processor(
                json._str_impl.bind_processor(dialect), test
            )

        def process_result_value(self, value, dialect):
            return self.type_adapter.validate_python(value)

        @memoized_property
        def type_adapter(self):
            return TypeAdapter(self._type)

        def copy(self, *a, **kw):
            return type(self)(self._type, *a, **kw)

    if not TYPE_CHECKING:

        class BigInt(int):
            pass

    def get_sqlalchemy_type(field: FieldInfo) -> Any:
        extra = (
            field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
        )
        if extra.get("json_in_sql") == "true":
            return PydanticJSON(field.annotation)

        if field.annotation is None:
            return _get_sqlalchemy_type(field)
        root_type = field.annotation

        try:
            if get_origin(root_type) == ModelMapped:
                root_type = get_args(root_type)[0]
            if (
                get_origin(root_type) == UnionType
                or get_origin(root_type) == typing.Union
            ):
                args = [
                    arg
                    for arg in get_args(root_type)
                    if not (isinstance(arg, NoneType) or arg == NoneType or arg is None)
                ]
                if len(args) == 1:
                    root_type = args[0]
                    if hasattr(field, "nullable"):
                        field.nullable = True  # type: ignore
            if get_origin(root_type) == Annotated:
                root_type = get_args(root_type)[0]

            if (
                get_origin(root_type) == UnionType or issubclass(root_type, BaseModel)
            ) and extra.get("json_in_sql") != "false":
                return PydanticJSON(field.annotation)
            if issubclass(root_type, UUID):
                return UUIDType
            if issubclass(root_type, BigInt):
                return BigInteger
        except TypeError:
            if get_origin(root_type) == Literal:
                root_type = type(get_args(root_type)[0])

        original_annotation = field.annotation
        try:
            field.annotation = root_type
            return _get_sqlalchemy_type(field)
        finally:
            field.annotation = original_annotation

    import sqlmodel.main as sqlmodel_main

    sqlmodel_main.get_sqlalchemy_type = get_sqlalchemy_type
    import sqlmodel.main as sqlmodel_main

    sqlmodel_main.get_sqlalchemy_type = get_sqlalchemy_type


tx_lock = anyio.Lock()

session_context = Context[AsyncSession]()


class ContextSession:
    ctx = CTX()
    session: AsyncSession | None = None
    lock: anyio.Lock | None = None
    commit_hooks: list[Callable[[], Awaitable]]
    rollback_hooks: list[Callable[[], Awaitable]]

    async def __aenter__(self):
        if (root := self.ctx.get(None)) is not None:
            self.context = None
            return root
        if PluginConfig.parallel is False:
            await tx_lock.__aenter__()
            self.lock = tx_lock
        if self.session is not None:
            self.context = session_context.use(self.session)
            self.context.__enter__()
            self.self_context = self.ctx.use()
            self.self_context.__enter__()
            return self
        lazy = LazySession.ctx.get(None)
        if lazy is not None:
            self.context = None
            session = await lazy.start()
            return session
        await self.start()
        assert self.session is not None
        self.context = session_context.use(self.session)
        self.context.__enter__()
        self.self_context = self.ctx.use()
        self.self_context.__enter__()
        return self

    async def start(self):
        self.session = Dependencies.ctx.get().resolve(AsyncSessionmaker)()
        self.commit_hooks = []
        self.rollback_hooks = []

    async def __aexit__(self, exc_type, exc_value, trace):
        if self.lock is not None:
            await self.lock.__aexit__(exc_type, exc_value, trace)

        if not self.context:
            return
        self.self_context.__exit__(exc_type, exc_value, trace)
        self.context.__exit__(exc_type, exc_value, trace)
        if self.session is None:
            return
        if not exc_type:
            try:
                await self.session.commit()
            finally:
                await self.session.__aexit__(exc_type, exc_value, trace)
            for hook in self.commit_hooks:
                await hook()
        else:
            try:
                await self.session.rollback()
            finally:
                await self.session.__aexit__(exc_type, exc_value, trace)
            for hook in self.rollback_hooks:
                await hook()

    async def commit(self):
        assert self.session is not None
        await self.session.commit()

    async def rollback(self):
        assert self.session is not None
        await self.session.rollback()


class LazySession:
    ctx = CTX()
    session: ContextSession | None = None

    async def start(self):
        if self.session:
            return self.session
        logger.trace("lazy session started")
        self.session = ContextSession()
        await self.session.start()
        await self.session.__aenter__()
        return self.session

    async def __aenter__(self):
        if self.ctx.get(None) is not None:
            self.context = None
            return
        self.context = use_context_value(self.ctx, self)
        self.context.__enter__()
        return self

    async def __aexit__(self, exc_type, exc_value, trace):
        if not self.context:
            return
        self.context.__exit__(exc_type, exc_value, trace)
        if not self.session:
            return
        await self.session.__aexit__(exc_type, exc_value, trace)


class SelectOfScalarExtended[TM](SelectOfScalar[TM]):
    inherit_cache = True

    def __await__(self) -> typing.Generator[Any, Any, ScalarResult[TM]]:
        return self.exec().__await__()

    async def exec(self) -> ScalarResult[TM]:
        session = session_context.get()
        return await session.exec(self)  # type: ignore

    async def stream(self) -> typing.Sequence[TM]:
        return (await self).all()

    async def all(self) -> typing.Sequence[TM]:
        return (await self).all()

    async def all_map[T](self, callback: Callable[[TM], T]):
        return list(map(callback, await self.all()))

    async def unique(self, strategy: Any | None = None) -> ScalarResult[TM]:
        return (await self).unique(strategy)

    async def partitions(self, size: int | None = None) -> typing.Iterator[typing.Sequence[TM]]:
        return (await self).partitions(size)

    async def fetchall(self) -> typing.Sequence[TM]:
        return (await self).fetchall()

    async def fetchmany(self, size: int | None = None) -> typing.Sequence[TM]:
        return (await self).fetchmany(size)

    async def first(self) -> TM | NoneType:
        return (await self).first()

    async def one(self) -> TM:
        return (await self).one()

    async def one_or_none(self) -> TM | NoneType:
        return (await self).one_or_none()

    async def one_or_raise(self, exception: BaseException) -> TM:
        result = await self.one_or_none()
        if result is None:
            raise exception
        return result

    async def one_or_call[T](self, callback: Callable[[], T]) -> T | TM:
        result = await self.one_or_none()
        if result is None:
            return callback()
        return result

    async def one_or[T](self, default: T) -> T | TM:
        result = await self.one_or_none()
        if result is None:
            return default
        return result


def fk(column: Any):
    column = col(column)
    return f"{column.table.name}.{column.name}"


def context_transaction(standalone: bool = False):
    """Supplies session to context. standalone - create new session"""

    def wrapper[**P, T](cb: Callable[P, Awaitable[T]]):  # type:ignore
        @asynccontextmanager
        async def run(*args: P.args, **kwargs: P.kwargs):
            session_provider = LazySession.ctx.get(None)

            if session_provider and not standalone:
                session = await session_provider.start()
                assert session.session is not None
                with use_context_value(session_context, session.session):
                    yield await cb(*args, **kwargs)
                return

            if (session_context.get(None) is not None) and not standalone:
                yield await cb(*args, **kwargs)
                return

            async with ContextSession():
                yield await cb(*args, **kwargs)

        return run

    return wrapper


def transaction(tries: int = 3, standalone: bool = False):  # /NOSONAR
    tx = context_transaction(standalone)
    delays = [0] + [min(i**1.5, 60) for i in range(tries)]

    def wrapper[**P, T](
        cb: Callable[P, Awaitable[T]],
    ) -> Callable[P, Awaitable[T]]:
        run_tx = tx(cb)

        @wraps(cb)
        async def wrapped(*args: P.args, **kwargs: P.kwargs):
            if session_context.get(None) is not None and not standalone:
                return await cb(*args, **kwargs)

            pass_exception = False

            for i in range(tries - 1):
                try:
                    async with run_tx(*args, **kwargs) as result:
                        return result

                except Exception as e:
                    if pass_exception:
                        raise e
                    logger.error(e)
                    await anyio.sleep(delays[i])
                    continue

            async with run_tx(*args, **kwargs) as result:
                return result

        wrapped.__sql_plugin_tx_fn__ = True  # type: ignore

        return wrapped

    return wrapper


@plugin.setup(stage=-20, dependencies=[patch_conventions])
def setup_engine(config: ConnectionConfig.Value) -> AsyncEngine:
    engine = create_async_engine(
        config.url,
        **config.model_dump(
            exclude_defaults=True,
            exclude={"kwargs", "no_multithread", "expire_on_commit", "url"},
        ),
        **config.kwargs,
    )
    LifecycleModule.get().on_stop(engine.dispose)
    if not config.allow_multithread and PluginConfig.parallel is None:
        logger.warning("Parallel transactions are disabled application wide")
        PluginConfig.parallel = False
    return engine


AsyncSessionmaker = async_sessionmaker[AsyncSession]


@plugin.setup(stage=-20)
def sessionmaker_setup(
    engine: setup_engine.Result, config: ConnectionConfig.Value
) -> AsyncSessionmaker:
    return async_sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=config.expire_on_commit,
        class_=AsyncSession,
    )


@plugin.setup(stage=-20, dependencies=[TypeRef(type=AsyncSessionmaker)])
@transaction()
async def test_connection():
    session = session_context.get()
    await session.exec(select(1))


def get_sessionmaker():
    return Dependencies.ctx.get().resolve(AsyncSessionmaker)


no_alembic_dependencies = simple_plugin(bind=False, load=False)


@no_alembic_dependencies.setup()
async def create_all(session: AsyncSessionmaker):
    async with session.begin() as conn:
        await conn.run_sync(fn=SQLModel.metadata.create_all)


@plugin.add_linker
def migrations_linker(deps: Plugin):
    cfg = config(MigrationConfig)

    if cfg.disable:
        return

    if cfg.alembic is None:
        deps.add(no_alembic_dependencies.rebuild())
    else:
        from .ext.alembic_migrations import plugin

        deps.add(plugin.rebuild())
