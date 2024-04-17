from contextlib import asynccontextmanager
from functools import wraps
from inspect import isfunction
from tempfile import TemporaryDirectory
from typing import Awaitable, Callable, Sequence, Type, Union
from rewire.config import ConfigModule
from rewire.dependencies import (
    Dependencies,
    DependenciesModule,
    InjectedDependency,
    Dependable,
)
from rewire.lifecycle import LifecycleModule
from rewire.space import Module, Space
from rewire_sqlmodel import SQLModel, transaction, plugin
from rewire_sqlmodel.ext.alembic_migrations import upgrade_db


class TemporalDBModule(Module, register=False):
    data: Sequence[Union[Callable[[], None], SQLModel]]

    @asynccontextmanager
    async def use_module(self):
        with TemporaryDirectory() as dir:
            ConfigModule.get().patch(
                {
                    "rewire_sqlmodel": {
                        "alembic": {"generate": False},
                        "url": f"sqlite+aiosqlite:///{dir}/db.sqlite",
                    }
                }
            )
            yield

    @transaction(1)
    async def _init(self):
        for item in self.data:
            if isfunction(item):
                item()
            else:
                item.add()

    def dependencies(self):
        dependencies = Dependencies()
        dependency = InjectedDependency.from_function(self._init)
        dependency.dependencies.append(upgrade_db)
        dependencies.bind(dependency)
        return dependencies


def prefill_db(
    *data: Union[Callable[[], None], SQLModel],
    modules: list[Type[Module]] = [DependenciesModule, ConfigModule, LifecycleModule],
    dependencies: list[Dependencies | Dependable] = [],
):
    def wrapper[**P, T](cb: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(cb)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            db = TemporalDBModule(data=data)
            async with Space(only=modules).add(db).init().use():
                await (
                    DependenciesModule.get()
                    .add(plugin)
                    .add(*dependencies)
                    .rebuild(inplace=True)
                    .add(TemporalDBModule.get().dependencies())
                    .solve()
                )
                async with LifecycleModule.get().use_running():
                    return await cb(*args, **kwargs)

        return wrapped

    return wrapper
