# Aare Forecasting

I bi vou binech das [momentan z viu AI-Hype ume isch](https://ai.aare.guru/) aber weisch wi geil we dr Aare Guru
scho paar Täg im vorus ungfähr wüsst wi warm d Aare de isch u wiviu Wasser si de het?

**Namensidee:** Aare Oraku \
**Logoidee**: Wätterfrosch wo uf sire Leitere steit und id witi luegt aus wärs ä Schiffsusguck während er mit sim Glas d Aare ab tribt <- di Idee hani vorem Name gha

Implementiert isch no nid viu aber ha dänkt i fa mau a.

Interessanterwiis gits scho ä Vorhersag zum Wasserabfluss (d Wassermängi im Aare.guru) vom Bund, i nime a di isch
scho sehr guet u chönnt direkt verwendet werde. Übrig blibt auso d Temperaturvorhersag, wo vermuetlech eifacher isch
aus d Abflussvorhersag, drum probiere mrs eis.

## Setup

Runs best with [uv](https://docs.astral.sh/uv/) and relies heavily on [DVC](https://dvc.org/). Only tested on Linux with bash and zsh.

```bash
uv venv --seed
uv sync
source .venv/bin/activate
```

To run the pipelines, do this (from the project root). These currently just calculate the baselines.

```bash
export PYTHONPATH="$PWD"
dvc repro
```

To run the notebooks, make sure to set the kernel to the lokal venv (`.venv/bin/python`) in your IDE.
In vscode, when it asks after first time execution, set the environment to the recommended ".venv" env.
Also, I suggest launching vscode from the console after executing `export PYTHONPATH="$PWD"` because unlike
PyCharm, it doesn't autodetect the `aare` package. Can also be [done in code](https://stackoverflow.com/a/16114586/10883465) if you must.

If you want to use JupyterLab, you need to install the kernelspec first: `python -m ipykernel install --user --name aare-forecast` \
Then set the PYTHONPATH again and run `jupyter lab`. You'll need to change the kernel in the top right for every notebook.

## Quellen & Links

D Date chöme vom [aare.guru](https://aare.guru) säuber oder vom [BAFU](https://www.hydrodaten.admin.ch/),
aggregiert im wahnsinnig tolle Archiv vom [Bureau für digitale Existenz](https://bureau.existenz.ch/).

- [Aare.guru API Description](https://aareguru.existenz.ch/)
- [InfluxDB Access ❤️](https://api.existenz.ch/#influx)
- [SwissMetNet Dataset Description](https://api-datasette.konzept.space/existenz-api/smn_parameters)
- [BAFU Hydrologie Dataset Description](https://api-datasette.konzept.space/existenz-api/hydro_parameters)
- [Hydrologische Vorhersagen vom Bund (nüt dopplet mache)](https://www.bafu.admin.ch/bafu/de/home/themen/wasser/extremereignisse/hydrologische-vorhersagen-des-bundes.html)

### Potenzielle Ressourcen

- http://www.watercenter.org/physical-water-quality-parameters/water-temperature/water-temperature-ranges-in-rivers-and-streams/
- https://storymaps.arcgis.com/stories/ef8b1542a6c0411ba96779b26151e399
- https://pmc.ncbi.nlm.nih.gov/articles/PMC5994338/
- https://agupubs.onlinelibrary.wiley.com/doi/10.1029/98WR01877
- https://hess.copernicus.org/articles/25/2951/2021/
- https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2427.2006.01597.x

## Lizenz

AGPL-3.0

(für aare.guru müesst das nid kümmere, i mache ds hie houptsächläch füre Spass u fänds riise geil 's i aare.guru z integriere wes de ändlech fr öppis isch)
