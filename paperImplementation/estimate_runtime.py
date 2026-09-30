"""
estimate_runtime.py

Gyors, önálló futásidő-becslő a build_cost_matrix.py-ban lévő
build_offline_cost_matrix() teljes lefuttatása ELŐTT.

Egyetlen VALÓS SARIMAX fit-et végez el a legnagyobb várható tanítási
ablakon (ami a te konkrét paraméterezésed - min_train_days, max_train_days,
t_prime_step_days - mellett ténylegesen elő fog fordulni a teljes futás
során), és ebből ad egy optimista/pesszimista becslést a teljes futásidőre.

Használat (a te 4 éves órás Series-eden):

    from estimate_runtime import estimate_runtime

    estimate_runtime(
        series,
        order=(1, 1, 1),
        min_train_days=14,
        horizon_days=30,
        max_train_days=90,      # rolling window cap - ld. build_cost_matrix.py
        t_prime_step_days=7,    # heti t' mintavétel
    )

Ez csak EGY fit-et csinál, tehát pár másodperc/perc alatt lefut - nem kell
hozzá a teljes 4 éves pipeline-t elindítanod, hogy tudd, mennyi időbe fog
telni.
"""

import time
import argparse

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from build_cost_matrix import create_exog, load_data, TARGET, _make_synthetic_series


def estimate_runtime(
    series: pd.Series,
    order=(1, 1, 1),
    min_train_days: int = 7,
    horizon_days: int = 30,
    max_train_days: int | None = None,
    t_prime_step_days: int = 1,
    fit_kwargs: dict | None = None,
) -> dict:
    """
    Egyetlen VALÓS fit lemérésével (a legnagyobb várható ablakméreten) ad egy
    durva alsó és felső becslést a build_offline_cost_matrix() teljes
    futásidejére.

    Az "optimista" becslés azt feltételezi, hogy minden fit kb. ugyanannyi
    ideig tart, mint ez az egy mért fit (ez volt megfigyelhető egy kis
    teszt-adaton). A "pesszimista" becslés azt feltételezi, hogy a fit ideje
    kb. arányos a tanítási ablak méretével (ez a reálisabb határeset nagy,
    expanding - vagyis max_train_days=None melletti - ablakoknál).

    Paraméterek
    ----------
    series : pd.Series
        Órás bontású célváltozó, DatetimeIndex-szel - ugyanaz, amit a
        build_offline_cost_matrix() is kap majd.
    order, min_train_days, horizon_days, max_train_days, t_prime_step_days :
        Pontosan ugyanazok a paraméterek, amiket a tényleges
        build_offline_cost_matrix() híváshoz tervezel használni - a becslés
        csak akkor pontos, ha ugyanazokat adod meg itt is.
    fit_kwargs : dict
        Extra kwargs a model.fit()-hez (pl. {"maxiter": 100}).

    Visszatérés
    ----------
    dict : n_rows, worst_case_fit_seconds, optimistic_total_seconds,
           pessimistic_total_seconds
    """

    fit_kwargs = fit_kwargs or {"maxiter": 100}

    all_days = pd.DatetimeIndex(sorted(series.index.normalize().unique()))

    if len(all_days) <= min_train_days:
        raise ValueError(
            f"Nincs elég nap az adatban ({len(all_days)}) a "
            f"min_train_days={min_train_days} tanítási ablakhoz."
        )

    forecastable_days = all_days[min_train_days:]
    T = len(forecastable_days)
    n_rows = len(range(0, T, t_prime_step_days))

    # a legnagyobb ablak, ami ténylegesen elő fog fordulni a futás során
    last_train_end = forecastable_days[-1]

    if max_train_days is not None:
        cutoff = last_train_end - pd.Timedelta(days=max_train_days)
        worst_case_train = series[(series.index >= cutoff) & (series.index < last_train_end)]
    else:
        worst_case_train = series[series.index < last_train_end]

    print(f"T={T} nap, ebből {n_rows} sor lenne kiszámolva "
          f"(t_prime_step_days={t_prime_step_days}).")
    print(f"Referencia-fit mérése a legnagyobb várható ablakon "
          f"({len(worst_case_train)} óra, {len(worst_case_train)/24:.0f} nap)...")

    exog = create_exog(worst_case_train.index)
    model = SARIMAX(worst_case_train, exog=exog, order=order, enforce_invertibility=False)

    t0 = time.perf_counter()
    model.fit(disp=False, **fit_kwargs)
    worst_case_fit_seconds = time.perf_counter() - t0

    print(f"  -> ez az egy fit {worst_case_fit_seconds:.1f} másodpercig tartott.\n")

    optimistic_total = n_rows * worst_case_fit_seconds
    # pesszimista: lineáris skálázást feltételezve az átlagablak kb. fele a
    # legnagyobbnak, ezért az átlagos fit ~fele ennyi ideig tart - de
    # n_rows sorra összegezzük
    pessimistic_total = n_rows * (worst_case_fit_seconds / 2)

    print(f"Becsült sorok száma (fitek): {n_rows}")
    print(f"Optimista becslés (konstans fit-idő):    {optimistic_total/60:.1f} perc")
    print(f"Pesszimista becslés (lineáris skálázás): {pessimistic_total/60:.1f} perc")
    print("(A valóság valahol e kettő között lesz - a pontos érték a te "
          "gépeden/adatodon dől el, ez csak tájékozódásra való. A "
          "forecast/append lépések idejét ez a becslés nem tartalmazza, "
          "azok jellemzően elhanyagolhatók a fit idejéhez képest.)")

    return {
        "T": T,
        "n_rows": n_rows,
        "worst_case_fit_seconds": worst_case_fit_seconds,
        "optimistic_total_seconds": optimistic_total,
        "pessimistic_total_seconds": pessimistic_total,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Futásidő-becslés a build_offline_cost_matrix()-hoz, "
                     "InfluxDB-adaton (vagy --demo esetén szintetikus adaton)."
    )
    parser.add_argument("--demo", action="store_true",
                         help="Szintetikus adaton fut (nincs InfluxDB kapcsolat).")
    parser.add_argument("--start", type=str, default=None,
                         help="InfluxDB lekérdezés kezdete, pl. 2020-01-01")
    parser.add_argument("--end", type=str, default=None,
                         help="InfluxDB lekérdezés vége, pl. 2024-01-01")
    parser.add_argument("--min-train-days", type=int, default=14)
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--max-train-days", type=int, default=90)
    parser.add_argument("--t-prime-step-days", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()

    if args.demo or not (args.start and args.end):

        print("=" * 70)
        print("DEMO MÓD: szintetikus adat (nincs megadva --start/--end)")
        print("=" * 70)

        series = _make_synthetic_series(n_days=120, seed=42)

    else:

        print("=" * 70)
        print(f"INFLUXDB ADAT: {args.start} -> {args.end}")
        print("=" * 70)

        df = load_data(args.start, args.end)
        print(f"Lekérdezett sorok: {len(df)}\n")

        series = df[TARGET]

    estimate_runtime(
        series,
        order=(1, 1, 1),
        min_train_days=args.min_train_days,
        horizon_days=args.horizon_days,
        max_train_days=args.max_train_days,
        t_prime_step_days=args.t_prime_step_days,
    )
