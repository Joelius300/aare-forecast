# Alles im Rahmen vom Möglichen

- Gewichtete Evaluation, Apr-Mai könnte 1.5x zählen, Mai-Jul 2x (Saisonstart), Jul-Okt wieder 1.5x und Okt-Apr normal 1x
- Sample weights, für jährliche Saison (siehe oben) aber auch daily, weil die Nacht interessiert uns deutlich weniger als 12:00-20:00 z.B.
- Prediction Bands / Confidence Intervals, also Unsicherheiten wie bei MeteoSwiss
- See-Temperatur einbeziehen (<https://www.alplakes.eawag.ch/>)
- Weitere Messstationen (z.B. Thun) vorhersagen
- Muesch perf tests mache aber denke 1d partitions und witeri dimension uf run_ts macht Sinn fr forecast
- Teste söttsch ono materialized view fre forecast table wome effektiv wott abfrage (normalerwiis neuste pred run, aber distinct on fr aues miech äuä o sinn (immer di neusti pred pro Punkt))
  - Lug ono compression aa, so columnstore policy
- Unterschied zwischen predicted peak temp und actual peak temp als gute Metrik (absolute)
- Average diff zwischen pred peak und actual peak OHNE avg zeigt ob es einen bias hat zu über oder unterschätzen
- Fehler von MeteoTest vorherage irgendwie einbeziehen. Geht erst wenn wir mehr MeteoTest Daten haben. Probabilistische Vorhersage liefern sie glaube ich nie :/
