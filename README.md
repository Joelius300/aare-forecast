# Aare Forecasting

I bi vou binech das [momentan z viu AI-Hype ume isch](https://ai.aare.guru/) aber weisch wi geil we dr Aare Guru
scho ä Wuche im vorus chli wüsst wi warm d Aare de isch u wiviu Wasser si de het?

Implementiert isch no nid viu aber ha dänkt i fa mau a.

Interessanterwiis gits scho ä Vorhersag zum Wasserabfluss (d Wassermängi im Aare.guru) vom Bund, i nime a di isch
scho sehr guet. Zur Temperatur gits schiinbar keni, ds wär auso ä Challenge, aber mä cha o hingerfrage wieso dr Bund
keni macht (isches z schwär? äuä eher eifach belanglos fr si). Zur Azeig und o zur Vorhersag chönntme das auso direkt bruche.

**Logoidee**: Wätterfrosch wo uf sire Leitere steit und id witi luegt aus wärs ä Schiffsusguck während er mit sim Glas d Aare ab tribt

## Setup

Runs best with [uv](https://docs.astral.sh/uv/) and relies heavily on [DVC](https://dvc.org/).

```bash
uv venv
source .venv/bin/activate
```

To run the pipelines, do this (from root).

```bash
export PYTHONPATH="$PWD"
dvc repro
```

## Links

- [Aare.guru API Description](https://aareguru.existenz.ch/)
- [InfluxDB Access ❤️](https://api.existenz.ch/#influx)
- [SwissMetNet Dataset Description](https://api-datasette.konzept.space/existenz-api/smn_parameters)
- [BAFU Hydrologie Dataset Description](https://api-datasette.konzept.space/existenz-api/hydro_parameters)
- [Hydrologische Vorhersagen vom Bund (nüt dopplet mache)](https://www.bafu.admin.ch/bafu/de/home/themen/wasser/fachinformationen/zustand-der-gewaesser/hydrologische-vorhersagen-des-bundes.html)

### Potenzielle Ressourcen

- http://www.watercenter.org/physical-water-quality-parameters/water-temperature/water-temperature-ranges-in-rivers-and-streams/
- https://storymaps.arcgis.com/stories/ef8b1542a6c0411ba96779b26151e399
- https://pmc.ncbi.nlm.nih.gov/articles/PMC5994338/

## Lizenz

AGPL-3.0

(für aare.guru fände mr sicher ä gueti Lösig, i mache ds hie houptsächläch füre Spass u fänds riise geil 's i aare.guru z integriere wes de ändlech fr öppis isch)
