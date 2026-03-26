import modal

volume = modal.Volume.from_name("surfdock-runs")

app = modal.App(
    "surfdock-tensorboard",
    image=modal.Image.debian_slim(python_version="3.12").pip_install("tensorboard>=2.20.0", "setuptools<82"),
)


class VolumeMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        if environ.get("PATH_INFO") in ["/", "/modal-volume-reload"]:
            try:
                volume.reload()
            except Exception:
                pass
        return self.wsgi_app(environ, start_response)


@app.function(
    volumes={"/runs": volume},
    max_containers=1,
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=100)
@modal.wsgi_app()
def tensorboard_app():
    import tensorboard

    board = tensorboard.program.TensorBoard()
    board.configure(logdir="/runs/workdir")
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    return tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
        experimental_middlewares=[VolumeMiddleware],
    )
