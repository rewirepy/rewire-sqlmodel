"""Usage

python -m rewire_sqlmodel.dev.merge {name}
"""

import sys
from os import listdir, remove, rename
from tempfile import TemporaryDirectory

import anyio

from alembic.config import Config as AlembicConfig
from alembic.script.base import ScriptDirectory
from rewire.config import ConfigModule
from rewire.dependencies import Dependencies
from rewire.lifecycle import LifecycleModule
from rewire.loader import LoaderModule
from rewire.log import LoggerModule
from rewire.plugins import simple_plugin
from rewire.space import Space
from rewire_sqlmodel.ext.alembic_migrations import upgrade_db

plugin = simple_plugin(load=False)


def sanitize(data: str):
    old = data
    data = "".join(
        x
        if x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz1234567890"
        else "_"
        for x in data
    )
    while old != data:
        old = data
        data = data.replace("__", "_")

    return data


@plugin.setup()
async def detect_rev(cfg: AlembicConfig):
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_revisions("heads")
    assert len(heads), "more than one head detected"
    head = heads[0]
    assert head is not None, "Ho heads"
    return head.revision


@plugin.setup(dependencies=[detect_rev])
def disable_migrations():
    for file in listdir("./alembic/versions"):
        if not file.endswith("dev.py"):
            continue
        rename(f"./alembic/versions/{file}", f"./alembic/versions/{file}.tmp")


@plugin.setup(dependencies=[upgrade_db])
def cleanup(rev: detect_rev.Result):
    name = (sys.argv + ["dev"] * 2)[1]
    sanitized_name = sanitize(name)

    for file in listdir("./alembic/versions"):
        if file.endswith("dev.py"):
            rename(
                f"./alembic/versions/{file}",
                f"./alembic/versions/{rev}_{sanitized_name}.py",
            )
            file = f"./alembic/versions/{rev}_{sanitized_name}.py"

            with open(file) as f:
                data = f.read()

            new_rev = data.split("revision = '")[1].split("'")[0]

            data = data.replace("Dev", name)
            data = data.replace(new_rev, rev)
            with open(file, "w") as f:
                f.write(data)

        if file.endswith(".tmp"):
            remove(f"./alembic/versions/{file}")


async def run():
    with TemporaryDirectory() as dir:
        async with Space(
            only=[
                LifecycleModule,
                LoaderModule,
                ConfigModule,
                LoggerModule,
            ]
        ).init().add().use():
            ConfigModule.get().patch(
                {
                    "rewire_sqlmodel": {
                        "alembic": {"generate": True},
                        "url": f"sqlite+aiosqlite:///{dir}/db.sqlite",
                    }
                }
            )
            await LoaderModule.get().discover().load()

            from rewire_sqlmodel import plugin as sql_plugin
            from rewire_sqlmodel.ext.alembic_migrations import (
                plugin as alembic_plugin,
            )
            from rewire_sqlmodel.ext.alembic_migrations import (
                pre_upgrade_db,
            )

            deps = (
                Dependencies()
                .add(plugin)
                .add(sql_plugin)
                .add(alembic_plugin)
                .rebuild()
                .indexed()
            )
            pre_upgrade_db.resolve(deps)._dependencies.append(
                disable_migrations.resolve(deps)
            )
            await deps.solve()
            await LifecycleModule.get().stop()


if __name__ == "__main__":
    anyio.run(run)
