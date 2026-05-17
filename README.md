# Aare Oraku

**Status**: [MVP auf aare.guru](https://aare.guru/)

I bi vou binech das [momentan z viu AI-Hype ume isch](https://ai.aare.guru/) aber weisch wi geil we dr Aare Guru
scho paar Täg im vorus ungfähr wüsst wi warm d Aare de isch u wiviu Wasser si de het?

Füre Bevölkerigsschutz gits scho ä Vorhersag zum Wasserabfluss (d Wassermängi im Aare.guru) vom Bund, i nime a di isch
scho sehr guet u chönnt direkt verwendet werde. Übrig blibt auso d Temperaturvorhersag, wo vermuetlech eifacher isch
aus d Abflussvorhersag, drum probiere mrs eis :)

## Setup

Relies on [uv](https://docs.astral.sh/uv/) and [DVC](https://dvc.org/). Uses [just](https://just.systems/) to simplify commands.
Only tested on Linux with bash and zsh.

First [install uv](https://docs.astral.sh/uv/getting-started/installation/), then use it to install just with:

```bash
uv tool install rust-just
```

Then creating the Python environment is as simple as

```bash
just sync
```

To run the pipelines, do this. These currently just calculate the baselines.

```bash
just repro
```

To run the notebooks, make sure to set the kernel to the lokal venv (`.venv/bin/python`) in your IDE.
In vscode, when it asks after first time execution, set the environment to the recommended ".venv" env.

If you want to use JupyterLab, you need to install the kernelspec first: `just install-kernel` \
Then run `jupyter lab`. You'll need to change the kernel in the top right for every notebook.

To start the mlflow ui and the optuna dashboard for experiment tracking and hyperparameter tuning, use the following commands in separate terminals:

```bash
just mlflow
just optuna
```

Or use

```bash
just track
```

If you want to work with the code instead of just running it, I suggest you install `nbstripout` (already in venv) and `prek` (`uv tool install prek`).
To set them up, run the following.

```bash
uv run nbstripout --install --attributes .gitattributes
prek install -f
```

## Quellen & Links

D Date chöme vom [aare.guru](https://aare.guru) säuber oder vom [BAFU](https://www.hydrodaten.admin.ch/),
aggregiert im wahnsinnig tolle Archiv vom [Bureau für digitale Existenz](https://bureau.existenz.ch/).

- [Aare.guru API Description](https://aareguru.existenz.ch/)
- [InfluxDB Access ❤️](https://api.existenz.ch/#influx)
- [SwissMetNet Dataset Description](https://api-datasette.konzept.space/existenz-api/smn_parameters)
- [BAFU Hydrologie Dataset Description](https://api-datasette.konzept.space/existenz-api/hydro_parameters)
- [Hydrologische Vorhersagen vom Bund (nüt dopplet mache)](https://www.bafu.admin.ch/de/hydrologische-vorhersagen-und-warnungen)

## Lizenz

AGPL-3.0

The deployed API has additional usage restrictions as you can see when you visit it.