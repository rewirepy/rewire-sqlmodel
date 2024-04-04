from contextlib import suppress
from rewire.config import ConfigDependency
from rewire.plugins import simple_plugin

with suppress(ImportError):
    from fastapi import FastAPI  # type: ignore
    from fastapi.routing import APIRoute  # type: ignore
    from rewire_sqlmodel import transaction
    from starlette.routing import request_response  # type: ignore

    plugin = simple_plugin()

    @plugin.bind
    class Config(ConfigDependency):
        transaction_tries: int = 3

    @plugin.setup(priority=1000, stage=100)
    def monkey_patch_fastapi(app: FastAPI, config: Config.Value):
        for route in app.routes:
            if not isinstance(route, APIRoute):
                continue
            if not getattr(route.dependant.call, "__sql_plugin_tx_fn__", False):
                continue
            route.app = request_response(
                transaction(tries=config.transaction_tries)(route.get_route_handler())
            )

    monkey_patch_fastapi.optional = True
