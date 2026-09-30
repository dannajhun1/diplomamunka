"""
build_cost_matrix.py

Offline cost mátrix (C) felépítése SARIMA modellekre, a
cost-aware-retraining-algorithms repó (Mahadevan & Mathioudakis,
"Cost-Aware Retraining for Machine Learning", Knowledge-Based Systems 2024)
módszertana szerint:

    C[t', t] = annak a modellnek a hibája (MAE), amit a t' napon (újra)
               tanítottunk, és a t napi adaton értékelünk ki (t >= t').

A kulcsötlet, ami elkerüli a T^2 darab teljes model.fit()-et: soronként
(minden t'-re) csak EGYSZER tanítunk (model.fit), utána a sor mentén
jobbra haladva a statsmodels SARIMAX results.append(..., refit=False)
hívásával csak KITERJESZTJÜK a Kalman-szűrő állapotát a valós adatokkal,
paraméter-újraoptimalizálás nélkül. Ez pontosan azt szimulálja, mi történt
volna, ha a t' napon tanított modellt sosem tanítjuk újra, csak "élteti"
tovább a bejövő adat.

Költség: T darab model.fit() + soronként legfeljebb `horizon_days` darab
(olcsó) append+forecast lépés -> O(T) drága művelet, O(T * horizon) olcsó
művelet, NEM O(T^2) drága művelet.

A kimenet (a C mátrix + a hozzá tartozó napok listája) közvetlenül
átadható a cost-aware-retraining-algorithms repó dp.py / oracle.py
moduljainak (add_retraining_cost, dp_iterative, retrains_iterative).
"""

import os
import json
import time
import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from influxdb_client import InfluxDBClient  # pip install influxdb-client
from dotenv import load_dotenv  # pip install python-dotenv


# ============================================================
# .ENV BETÖLTÉSE / INFLUXDB KAPCSOLAT
# ============================================================
# Ugyanaz a séma, mint az incremental_learning.py-ban - a .env fájlt a
# jelenlegi munkakönyvtárból olvassa.

load_dotenv()

TARGET = "Global_active_power"

INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")
INFLUX_MEASUREMENT = os.getenv("INFLUX_MEASUREMENT")


def _check_influx_env():
    """Csak akkor hívjuk, ha ténylegesen InfluxDB-t akarunk lekérdezni -
    demo/szintetikus módban nem kell .env."""

    for var_name, var_value in [
        ("INFLUX_TOKEN", INFLUX_TOKEN),
        ("INFLUX_ORG", INFLUX_ORG),
        ("INFLUX_BUCKET", INFLUX_BUCKET),
        ("INFLUX_MEASUREMENT", INFLUX_MEASUREMENT),
    ]:
        if not var_value:
            raise RuntimeError(
                f"Hiányzó környezeti változó: {var_name} "
                f"(ellenőrizd a .env fájlt)"
            )


def load_data(start, end) -> pd.DataFrame:
    """
    Lekérdezi az adatokat InfluxDB-ből a [start, end) intervallumra,
    óránkénti átlaggal (aggregateWindow) - ugyanaz a lekérdezés, mint az
    incremental_learning.py-ban.

    Visszatérés: DataFrame, index=DatetimeIndex (naiv, UTC-ből lokalizálva),
    egyetlen TARGET nevű oszloppal.
    """

    _check_influx_env()

    start_str = pd.Timestamp(start).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = pd.Timestamp(end).strftime("%Y-%m-%dT%H:%M:%SZ")

    flux = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start_str}, stop: {end_str})
      |> filter(fn: (r) => r._measurement == "{INFLUX_MEASUREMENT}")
      |> filter(fn: (r) => r._field == "{TARGET}")
      |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
      |> keep(columns: ["_time", "_value"])
      |> sort(columns: ["_time"])
    '''

    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    try:
        tables = client.query_api().query(flux)
    finally:
        client.close()

    rows = [
        {"datetime": record.get_time(), TARGET: record.get_value()}
        for table in tables
        for record in table.records
    ]

    df = pd.DataFrame(rows)

    if df.empty:
        return df.set_index(pd.DatetimeIndex([], name="datetime"))

    df = df.set_index("datetime").sort_index()
    df.index = df.index.tz_localize(None)

    # FONTOS: a Flux lekérdezés createEmpty: false miatt a hiányzó órákat
    # egyszerűen kihagyja (nem NaN-sorként adja vissza), ami szabálytalan
    # (freq nélküli) indexet eredményez. A statsmodels SARIMAX .append()-je
    # viszont explicit, rögzített frekvenciájú, hézagmentes indexet vár -
    # enélkül "Given endog does not have an index that extends the index
    # of the model" hibát dob. Ezért itt kényszerítjük a teljes, szabályos
    # óránkénti rácsra, és a ténylegesen hiányzó órákat ffill-lel töltjük ki
    # (ugyanúgy, mint korábban, csak most a hiányzó IDŐPONTOKRA is, nem csak
    # a hiányzó ÉRTÉKEKRE).
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_index)
    df.index.name = "datetime"

    df = df.ffill().bfill()

    return df


# ============================================================
# EXOGÉN VÁLTOZÓK (ugyanaz a séma, mint az incremental_learning.py-ban)
# ============================================================

def create_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Óra + hét napja dummy-változók, ugyanúgy, mint a fő pipeline-ban."""

    data = pd.DataFrame(index=index)

    for hour in range(1, 24):
        data[f"hour_{hour}"] = (index.hour == hour).astype(int)

    for day in range(1, 7):
        data[f"dow_{day}"] = (index.dayofweek == day).astype(int)

    return data


# ============================================================
# OFFLINE COST MÁTRIX
# ============================================================

def build_offline_cost_matrix(
    series: pd.Series,
    order=(1, 1, 1),
    min_train_days: int = 7,
    horizon_days: int | None = 60,
    forecast_hours: int = 24,
    enforce_invertibility: bool = False,
    fit_kwargs: dict | None = None,
    max_train_days: int | None = None,
    t_prime_step_days: int = 1,
    verbose: bool = True,
):
    """
    Felépíti a C[t', t] offline cost mátrixot egy órás bontású, TARGET-et
    tartalmazó pandas Series-en (index: DatetimeIndex, óránkénti).

    Paraméterek
    ----------
    series : pd.Series
        Órás bontású célváltozó (pl. Global_active_power), DatetimeIndex-szel.
        Ugyanaz, amit a load_data() ad vissza (a TARGET oszlop).
    order : tuple
        SARIMA (p,d,q) rend.
    min_train_days : int
        Hány nap adatot használjunk a legelső (t'=0) modell tanításához,
        mielőtt az első forecastolható nap következik.
    horizon_days : int vagy None
        Egy adott t'-nél tanított modellt legfeljebb hány napra "sétáltatunk"
        előre (t - t' <= horizon_days). None esetén a teljes horizonton fut
        (T x T méretű mátrix - real-world adatnál drága lehet!).
    forecast_hours : int
        Hány órás előrejelzést készítünk naponta (alapból 24 = egy teljes nap).
    fit_kwargs : dict
        Extra kwargs a model.fit()-hez (pl. {"maxiter": 50}).
    max_train_days : int vagy None
        FONTOS SKÁLÁZÁSI PARAMÉTER. Ha meg van adva, minden sor tanítása
        csak a train_end előtti utolsó `max_train_days` napot használja
        (rolling/csúszó ablak), NEM a teljes eddigi historikus adatot
        (expanding window). Enélkül a fit mérete (és így valószínűleg a
        fit ideje is) a sorindexszel arányosan nő, ami az ÖSSZES fit
        együttes költségét O(T) helyett O(T^2)-té teheti nagy T-nél
        (pl. több éves adatnál). Ha None, expanding window (mint eddig).
    t_prime_step_days : int
        Csak minden `t_prime_step_days`-edik napra készítünk sort (t'
        subsampling) - pl. 7 esetén heti gyakorisággal. Ez lineárisan
        csökkenti a T (=fitek száma) értékét, a mátrix mérete (és a days
        lista) változatlan marad, csak a ki nem számolt sorok maradnak NaN.
        Módszertanilag védhető döntés, ha a driftdinamika lassabb, mint a
        napi léptékű felbontás (ezt érdemes a dolgozatban megindokolni).
    verbose : bool
        Naplózás.

    Visszatérés
    ----------
    C : np.ndarray, shape (T, T)
        A cost mátrix. C[i, j] = NaN, ha j < i, j - i > horizon_days, vagy
        ha az i. sor a t_prime_step_days subsampling miatt ki lett hagyva.
    days : list[pd.Timestamp]
        A mátrix sorai/oszlopai közötti indexeknek megfelelő napok listája
        (days[i] = a t'=i / t=i naphoz tartozó dátum).
    meta : dict
        Futási statisztikák (fit_seconds összesen, mátrix mérete, stb.) -
        hasznos a dolgozat "overhead" fejezetéhez.
    """

    fit_kwargs = fit_kwargs or {"maxiter": 50}

    all_days = pd.Series(series.index.normalize().unique()).sort_values()
    all_days = pd.DatetimeIndex(all_days)

    if len(all_days) <= min_train_days:
        raise ValueError(
            f"Nincs elég nap az adatban ({len(all_days)}) a "
            f"min_train_days={min_train_days} tanítási ablakhoz."
        )

    forecastable_days = all_days[min_train_days:]
    T = len(forecastable_days)

    if T == 0:
        raise ValueError("Nincs egyetlen forecastolható nap sem - ellenőrizd a bemenetet.")

    if horizon_days is None:
        horizon_days = T - 1

    C = np.full((T, T), np.nan)

    total_fit_seconds = 0.0
    total_append_seconds = 0.0
    total_forecast_calls = 0

    prev_fit_params = None  # az előző sor konvergált paraméterei -> warm start

    if verbose:
        n_rows = len(range(0, T, t_prime_step_days))
        window_note = (f"max {max_train_days} nap (rolling)" if max_train_days
                        else "expanding (teljes historikus ablak)")
        print(f"Cost mátrix építése: T={T} nap, ebből {n_rows} sor lesz kiszámolva "
              f"(t_prime_step_days={t_prime_step_days}), horizon={horizon_days} nap, "
              f"tanítási ablak: {window_note}")
        print(f"  -> kb. {n_rows} fit + kb. {n_rows * (horizon_days + 1)} forecast lépés")

    for i in range(0, T, t_prime_step_days):

        train_end = forecastable_days[i]

        # --- egyetlen "drága" lépés: a t'=i napon (újra)tanítjuk a modellt ---
        if max_train_days is not None:
            train_start_cutoff = train_end - pd.Timedelta(days=max_train_days)
            train = series[(series.index >= train_start_cutoff) & (series.index < train_end)]
        else:
            train = series[series.index < train_end]

        # A boolean-maszkos szeletelés elveszíti a .freq attribútumot még
        # egy amúgy hézagmentes órás indexen is - ezt itt visszaállítjuk,
        # mert e nélkül a SARIMAX .append() ValueError-t dob.
        train = train.asfreq("h").ffill()
        train_exog = create_exog(train.index)

        model = SARIMAX(
            train,
            exog=train_exog,
            order=order,
            enforce_invertibility=enforce_invertibility,
        )

        t0 = time.perf_counter()

        # Warm start az előző sor (t'=i-1) konvergált paramétereiből, ha van -
        # ez ugyanaz a trükk, mint a fő pipeline-ban, és jelentősen csökkenti
        # a ConvergenceWarning-ok esélyét, mert nem "nulláról" kell minden
        # sorban megtalálni az optimumot.
        if prev_fit_params is not None:
            results = model.fit(start_params=prev_fit_params, disp=False, **fit_kwargs)
        else:
            results = model.fit(disp=False, **fit_kwargs)

        prev_fit_params = results.params

        total_fit_seconds += time.perf_counter() - t0

        j_max = min(i + horizon_days, T - 1)

        current_results = results
        prev_day = train_end  # az utolsó nap, ameddig current_results "látott" adatot

        for j in range(i, j_max + 1):

            day_j = forecastable_days[j]
            day_j_end = day_j + pd.Timedelta(days=1)

            actual = series[(series.index >= day_j) & (series.index < day_j_end)]

            if len(actual) < forecast_hours:
                # nincs elég valós adat ehhez a naphoz (pl. az idősor vége) - kihagyjuk
                break

            if j > i:
                # olcsó lépés: kiterjesztjük a modellt az azóta eltelt valós adattal,
                # DE nem optimalizáljuk újra a paramétereket (refit=False)
                delta = series[(series.index >= prev_day) & (series.index < day_j)]
                delta = delta.asfreq("h").ffill()  # ld. fentebb - .freq visszaállítása
                delta_exog = create_exog(delta.index)

                t0 = time.perf_counter()
                current_results = current_results.append(
                    delta, exog=delta_exog, refit=False
                )
                total_append_seconds += time.perf_counter() - t0

                prev_day = day_j

            forecast_exog = create_exog(
                pd.date_range(start=day_j, periods=forecast_hours, freq="h")
            )

            forecast = current_results.forecast(steps=forecast_hours, exog=forecast_exog)
            total_forecast_calls += 1

            C[i, j] = mean_absolute_error(actual.iloc[:forecast_hours], forecast)

        if verbose and (i % max(1, T // 10) == 0 or i == T - 1):
            print(f"  [{i+1}/{T}] t'={train_end.date()} kész "
                  f"(eddig: fit={total_fit_seconds:.1f}s, append={total_append_seconds:.1f}s)")

    meta = {
        "T": T,
        "horizon_days": horizon_days,
        "min_train_days": min_train_days,
        "max_train_days": max_train_days,
        "t_prime_step_days": t_prime_step_days,
        "order": list(order),
        "total_fit_seconds": total_fit_seconds,
        "total_append_seconds": total_append_seconds,
        "total_forecast_calls": total_forecast_calls,
        "num_fits": len(range(0, T, t_prime_step_days)),
    }

    return C, list(forecastable_days), meta


# ============================================================
# MENTÉS / BETÖLTÉS
# ============================================================

def save_cost_matrix(C: np.ndarray, days: list, meta: dict, out_dir: str = "."):
    """Elmenti a C mátrixot (.npy), a napok listáját és a metaadatokat (.json)."""

    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "cost_matrix.npy"), C)

    with open(os.path.join(out_dir, "cost_matrix_days.json"), "w", encoding="utf-8") as f:
        json.dump([str(d.date()) for d in days], f)

    with open(os.path.join(out_dir, "cost_matrix_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Cost mátrix elmentve: {out_dir}/cost_matrix.npy "
          f"(shape={C.shape}), napok: {out_dir}/cost_matrix_days.json, "
          f"meta: {out_dir}/cost_matrix_meta.json")


def load_cost_matrix(out_dir: str = "."):
    """Visszatölti a save_cost_matrix által mentett C mátrixot, napokat, metaadatot."""

    C = np.load(os.path.join(out_dir, "cost_matrix.npy"))

    with open(os.path.join(out_dir, "cost_matrix_days.json"), "r", encoding="utf-8") as f:
        days = [pd.Timestamp(d) for d in json.load(f)]

    with open(os.path.join(out_dir, "cost_matrix_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    return C, days, meta


# ============================================================
# DEMO / ÖNTESZT SZINTETIKUS ADATTAL
# ============================================================

def _make_synthetic_series(n_days: int = 24, seed: int = 0) -> pd.Series:
    """
    Szintetikus, óránkénti "fogyasztás-szerű" adatsor: napi + heti szezonalitás,
    zaj, és egy szándékos szint-eltolódás (drift) a sorozat közepén - hogy a
    cost mátrixban lásson az ember érdemi mintázatot (a hiba nőjön a driftnél).
    """

    rng = np.random.default_rng(seed)

    index = pd.date_range("2024-01-01", periods=n_days * 24, freq="h")

    hour_effect = 1.0 + 0.6 * np.sin((index.hour - 6) / 24 * 2 * np.pi)
    dow_effect = 1.0 + 0.15 * (index.dayofweek >= 5).astype(float)

    drift = np.where(np.arange(len(index)) > len(index) // 2, 0.8, 0.0)

    noise = rng.normal(0, 0.08, size=len(index))

    values = hour_effect * dow_effect + drift + noise
    values = np.clip(values, 0.05, None)

    return pd.Series(values, index=index, name="Global_active_power")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Offline cost mátrix építése SARIMA-ra InfluxDB-adaton "
                     "(vagy --demo esetén szintetikus adaton)."
    )
    parser.add_argument("--demo", action="store_true",
                         help="Szintetikus adaton fut (nincs InfluxDB kapcsolat).")
    parser.add_argument("--start", type=str, default=None,
                         help="InfluxDB lekérdezés kezdete, pl. 2024-01-01")
    parser.add_argument("--end", type=str, default=None,
                         help="InfluxDB lekérdezés vége, pl. 2024-03-01")
    parser.add_argument("--min-train-days", type=int, default=14)
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--max-train-days", type=int, default=None,
                         help="Rolling window cap (ld. korábbi magyarázat) - "
                              "hosszabb (pl. több hónapos/éves) adatnál erősen ajánlott.")
    parser.add_argument("--t-prime-step-days", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="./cost_matrix_output")
    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()

    if args.demo or not (args.start and args.end):

        print("=" * 70)
        print("DEMO MÓD: szintetikus adat (nincs megadva --start/--end)")
        print("=" * 70)

        series = _make_synthetic_series(n_days=24, seed=42)

        min_train_days = 14
        horizon_days = 6
        max_train_days = args.max_train_days
        t_prime_step_days = args.t_prime_step_days

    else:

        print("=" * 70)
        print(f"INFLUXDB ADAT: {args.start} -> {args.end}")
        print("=" * 70)

        df = load_data(args.start, args.end)
        print(f"Lekérdezett sorok: {len(df)}")

        series = df[TARGET]

        min_train_days = args.min_train_days
        horizon_days = args.horizon_days
        max_train_days = args.max_train_days
        t_prime_step_days = args.t_prime_step_days

    C, days, meta = build_offline_cost_matrix(
        series,
        order=(1, 1, 1),
        min_train_days=min_train_days,
        horizon_days=horizon_days,
        forecast_hours=24,
        max_train_days=max_train_days,
        t_prime_step_days=t_prime_step_days,
        fit_kwargs={"maxiter": 100},
        verbose=True,
    )

    print()
    print(f"Kész. C mátrix alakja: {C.shape}")
    print(f"Napok: {days[0].date()} ... {days[-1].date()}")
    print(f"Összes fit idő: {meta['total_fit_seconds']:.2f} s "
          f"({meta['num_fits']} db fit)")
    print(f"Összes append idő: {meta['total_append_seconds']:.2f} s "
          f"({meta['total_forecast_calls']} db forecast)")

    print()
    print("Cost mátrix (kerekítve, csak a kitöltött sáv):")
    with np.printoptions(precision=3, suppress=True):
        print(C[:10, :10])

    save_cost_matrix(C, days, meta, out_dir=args.out_dir)
