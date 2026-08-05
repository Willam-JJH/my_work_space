from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import BarOverview, BaseDatabase, TickOverview
from vnpy.trader.datafeed import BaseDatafeed
from vnpy.trader.object import BarData, HistoryRequest, TickData


class FutureDataAccessError(ValueError):
    """Raised when date-aware code asks for rows after its bound date."""


class ReadOnlyDataError(RuntimeError):
    """Raised when code tries to mutate the read-only QClass data release."""


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    aliases: list[str]
    csmar_table_code: str | None
    relative_path: str
    format: str
    is_static: bool
    date_columns: list[str] = field(default_factory=list)
    primary_asof_column: str | None = None
    symbol_column: str | None = None
    column_descriptions: dict[str, dict[str, str]] = field(default_factory=dict)
    doc_paths: list[str] = field(default_factory=list)
    row_count: int | None = None
    source_row_count: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetSpec":
        return cls(
            dataset_id=data["dataset_id"],
            aliases=list(data.get("aliases", [])),
            csmar_table_code=data.get("csmar_table_code"),
            relative_path=data["relative_path"],
            format=data["format"],
            is_static=bool(data.get("is_static", False)),
            date_columns=list(data.get("date_columns", [])),
            primary_asof_column=data.get("primary_asof_column"),
            symbol_column=data.get("symbol_column"),
            column_descriptions=dict(data.get("column_descriptions", {})),
            doc_paths=list(data.get("doc_paths", [])),
            row_count=data.get("row_count"),
            source_row_count=data.get("source_row_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataCatalog:
    root: str | None
    datasets: list[DatasetSpec]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._index = self._build_index(self.datasets)

    @staticmethod
    def _build_index(datasets: Iterable[DatasetSpec]) -> dict[str, DatasetSpec]:
        index: dict[str, DatasetSpec] = {}
        for spec in datasets:
            for key in [spec.dataset_id, *spec.aliases]:
                if key in index and index[key].dataset_id != spec.dataset_id:
                    raise ValueError(f"duplicate dataset key: {key}")
                index[key] = spec
        return index

    @classmethod
    def load(cls, path: Path | str) -> "DataCatalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            root=data.get("root"),
            metadata=dict(data.get("metadata", {})),
            datasets=[
                DatasetSpec.from_dict(item)
                for item in data.get("datasets", [])
            ],
        )

    @classmethod
    def from_root(cls, root: Path | str) -> "DataCatalog":
        root_path = Path(root)
        catalog_path = root_path / "data_catalog.json"
        if catalog_path.exists():
            return cls.load(catalog_path)
        return scan_catalog(root_path)

    def list_datasets(self) -> list[str]:
        return sorted(spec.dataset_id for spec in self.datasets)

    def resolve(self, dataset_id_or_alias: str) -> DatasetSpec:
        try:
            return self._index[dataset_id_or_alias]
        except KeyError as exc:
            raise KeyError(f"unknown dataset: {dataset_id_or_alias}") from exc

    def describe(self, dataset_id_or_alias: str) -> DatasetSpec:
        return self.resolve(dataset_id_or_alias)

    def path_for(self, dataset_id_or_alias: str, root: Path | None = None) -> Path:
        spec = self.resolve(dataset_id_or_alias)
        base = root or (Path(self.root) if self.root else Path("."))
        return base / spec.relative_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "metadata": self.metadata,
            "datasets": [spec.to_dict() for spec in self.datasets],
        }


class QClassDataBundle:
    """Date-aware pandas access to raw `new_data` roots or generated releases."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.catalog = DataCatalog.from_root(self.root)

    def list_datasets(self) -> list[str]:
        return self.catalog.list_datasets()

    def describe(self, dataset_id_or_alias: str) -> DatasetSpec:
        return self.catalog.describe(dataset_id_or_alias)

    def path(self, dataset_id_or_alias: str) -> Path:
        return self.catalog.path_for(dataset_id_or_alias, self.root)

    def load_raw(
        self,
        dataset_id_or_alias: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        spec = self.describe(dataset_id_or_alias)
        return read_dataset(self.path(dataset_id_or_alias), spec, columns=columns)

    def at(self, date: str | datetime | pd.Timestamp) -> "QClassDatedData":
        return QClassDatedData(self, to_timestamp(date))


class QClassDatedData:
    """Bounded view with SickFun-compatible load/current/history/latest methods."""

    def __init__(self, bundle: QClassDataBundle, date: pd.Timestamp):
        self.bundle = bundle
        self.date = date.normalize()

    def load(
        self,
        dataset_id_or_alias: str,
        start: str | datetime | pd.Timestamp | None = None,
        end: str | datetime | pd.Timestamp | None = None,
        columns: list[str] | None = None,
        *,
        numeric_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        return self._load_bounded(
            dataset_id_or_alias,
            start=start,
            end=end,
            columns=columns,
            numeric_columns=numeric_columns,
            required_columns=[],
            project=True,
        )

    def current(
        self,
        dataset_id_or_alias: str,
        columns: list[str] | None = None,
        *,
        numeric_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.load(
            dataset_id_or_alias,
            start=self.date,
            end=self.date,
            columns=columns,
            numeric_columns=numeric_columns,
        )

    def history(
        self,
        dataset_id_or_alias: str,
        start: str | datetime | pd.Timestamp | None = None,
        lookback_days: int | None = None,
        columns: list[str] | None = None,
        *,
        numeric_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if start is not None and lookback_days is not None:
            raise ValueError("pass either start or lookback_days, not both")
        history_start = start
        if lookback_days is not None:
            if lookback_days < 0:
                raise ValueError("lookback_days cannot be negative")
            history_start = self.date - timedelta(days=lookback_days)
        return self.load(
            dataset_id_or_alias,
            start=history_start,
            end=self.date,
            columns=columns,
            numeric_columns=numeric_columns,
        )

    def latest(
        self,
        dataset_id_or_alias: str,
        by: str | None = None,
        columns: list[str] | None = None,
        *,
        numeric_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        spec = self.bundle.describe(dataset_id_or_alias)
        group_column = by or spec.symbol_column
        if spec.primary_asof_column is None:
            return self.load(
                dataset_id_or_alias,
                columns=columns,
                numeric_columns=numeric_columns,
            )

        required_columns = [group_column] if group_column is not None else []
        df = self._load_bounded(
            dataset_id_or_alias,
            columns=columns,
            numeric_columns=None,
            required_columns=required_columns,
            project=False,
        )
        if df.empty:
            return coerce_numeric_columns(project_columns(df, columns), numeric_columns)

        dates = date_series(df[spec.primary_asof_column])
        working = df.assign(__qclass_asof=dates).sort_values("__qclass_asof")
        if group_column is None:
            latest_date = working["__qclass_asof"].max()
            result = working[working["__qclass_asof"] == latest_date]
        else:
            result = working.groupby(group_column, as_index=False, sort=False).tail(1)
        projected = project_columns(
            result.drop(columns=["__qclass_asof"]).copy(),
            columns,
        )
        return coerce_numeric_columns(projected, numeric_columns)

    def _load_bounded(
        self,
        dataset_id_or_alias: str,
        start: str | datetime | pd.Timestamp | None = None,
        end: str | datetime | pd.Timestamp | None = None,
        columns: list[str] | None = None,
        numeric_columns: list[str] | None = None,
        required_columns: list[str] | None = None,
        project: bool = True,
    ) -> pd.DataFrame:
        spec = self.bundle.describe(dataset_id_or_alias)
        required = list(required_columns or [])
        if spec.primary_asof_column is not None:
            required.append(spec.primary_asof_column)
        read_columns = columns_with_required(columns, required)
        df = self.bundle.load_raw(dataset_id_or_alias, columns=read_columns)

        if spec.primary_asof_column is None:
            result = project_columns(df, columns) if project else df
            return coerce_numeric_columns(result, numeric_columns) if project else result

        start_date = to_timestamp(start) if start is not None else None
        end_date = to_timestamp(end) if end is not None else self.date
        if start_date is not None and start_date > self.date:
            raise FutureDataAccessError(
                f"start date {start_date.date()} is after bound date {self.date.date()}"
            )
        if end_date > self.date:
            raise FutureDataAccessError(
                f"end date {end_date.date()} is after bound date {self.date.date()}"
            )

        dates = date_series(df[spec.primary_asof_column])
        mask = dates <= end_date
        if start_date is not None:
            mask &= dates >= start_date
        filtered = df[mask].copy()
        result = project_columns(filtered, columns) if project else filtered
        return coerce_numeric_columns(result, numeric_columns) if project else result


class QClassVnpyDatafeed(BaseDatafeed):
    """Read-only vn.py datafeed backed by QClass `data_releases` or `new_data`."""

    def __init__(
        self,
        root: Path | str,
        *,
        dataset_id: str = "stock.daily_returns",
        gateway_name: str = "QCLASS",
        pricetick: float = 0.01,
    ):
        self.bundle = QClassDataBundle(root)
        self.dataset_id = dataset_id
        self.gateway_name = gateway_name
        self.pricetick = pricetick

    def init(self, output: Callable = print) -> bool:
        output(f"QClassVnpyDatafeed ready: {self.bundle.root}")
        return True

    def query_bar_history(
        self,
        req: HistoryRequest,
        output: Callable = print,
    ) -> list[BarData]:
        if req.interval not in (None, Interval.DAILY):
            output("QClass release adapter currently provides daily bars only.")
            return []
        frame, mapping = self._daily_frame(req.symbol, req.start, req.end)
        return [
            daily_row_to_bar(
                row,
                mapping,
                symbol=normalize_symbol(req.symbol),
                exchange=req.exchange,
                interval=req.interval or Interval.DAILY,
                gateway_name=self.gateway_name,
            )
            for _, row in frame.iterrows()
        ]

    def query_tick_history(
        self,
        req: HistoryRequest,
        output: Callable = print,
    ) -> list[TickData]:
        frame, mapping = self._daily_frame(req.symbol, req.start, req.end)
        return [
            daily_row_to_tick(
                row,
                mapping,
                symbol=normalize_symbol(req.symbol),
                exchange=req.exchange,
                gateway_name=self.gateway_name,
                pricetick=self.pricetick,
            )
            for _, row in frame.iterrows()
        ]

    def _daily_frame(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        end_value = end or datetime.now()
        mapping = daily_column_mapping(self.bundle, self.dataset_id)
        columns = list(dict.fromkeys(mapping.values()))
        data = self.bundle.at(end_value).load(
            self.dataset_id,
            start=start,
            end=end_value,
            columns=columns,
            numeric_columns=[
                mapping[key]
                for key in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "turnover",
                    "pre_close",
                    "limit_down",
                    "limit_up",
                ]
                if key in mapping
            ],
        )
        symbol_col = mapping["symbol"]
        date_col = mapping["date"]
        normalized = data[symbol_col].map(normalize_symbol)
        filtered = data[normalized == normalize_symbol(symbol)].copy()
        filtered["__qclass_date"] = date_series(filtered[date_col])
        filtered = filtered.sort_values("__qclass_date").drop(columns=["__qclass_date"])
        return filtered, mapping

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        req = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
            interval=interval,
        )
        return self.query_bar_history(req)

    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime,
    ) -> list[TickData]:
        req = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
            interval=Interval.DAILY,
        )
        return self.query_tick_history(req)


class QClassVnpyDatabase(BaseDatabase):
    """Read-only `BaseDatabase` facade for vn.py components that expect one."""

    def __init__(self, root: Path | str, **datafeed_kwargs: Any):
        self.datafeed = QClassVnpyDatafeed(root, **datafeed_kwargs)

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        return self.datafeed.load_bar_data(symbol, exchange, interval, start, end)

    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime,
    ) -> list[TickData]:
        return self.datafeed.load_tick_data(symbol, exchange, start, end)

    def save_bar_data(self, bars: list[BarData], stream: bool = False) -> bool:
        raise ReadOnlyDataError("QClassVnpyDatabase reads course releases; it does not save bars.")

    def save_tick_data(self, ticks: list[TickData], stream: bool = False) -> bool:
        raise ReadOnlyDataError("QClassVnpyDatabase reads course releases; it does not save ticks.")

    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
    ) -> int:
        raise ReadOnlyDataError("QClassVnpyDatabase is read-only.")

    def delete_tick_data(self, symbol: str, exchange: Exchange) -> int:
        raise ReadOnlyDataError("QClassVnpyDatabase is read-only.")

    def get_bar_overview(self) -> list[BarOverview]:
        frame, mapping = self._overview_frame()
        if frame.empty:
            return []
        return [
            BarOverview(
                symbol=symbol,
                exchange=infer_exchange(symbol),
                interval=Interval.DAILY,
                count=int(len(group)),
                start=group["__qclass_date"].min().to_pydatetime(),
                end=group["__qclass_date"].max().to_pydatetime(),
            )
            for symbol, group in frame.groupby("__qclass_symbol", sort=True)
        ]

    def get_tick_overview(self) -> list[TickOverview]:
        frame, mapping = self._overview_frame()
        if frame.empty:
            return []
        return [
            TickOverview(
                symbol=symbol,
                exchange=infer_exchange(symbol),
                count=int(len(group)),
                start=group["__qclass_date"].min().to_pydatetime(),
                end=group["__qclass_date"].max().to_pydatetime(),
            )
            for symbol, group in frame.groupby("__qclass_symbol", sort=True)
        ]

    def _overview_frame(self) -> tuple[pd.DataFrame, dict[str, str]]:
        mapping = daily_column_mapping(self.datafeed.bundle, self.datafeed.dataset_id)
        frame = self.datafeed.bundle.load_raw(
            self.datafeed.dataset_id,
            columns=[mapping["symbol"], mapping["date"]],
        )
        frame["__qclass_symbol"] = frame[mapping["symbol"]].map(normalize_symbol)
        frame["__qclass_date"] = date_series(frame[mapping["date"]])
        frame = frame.dropna(subset=["__qclass_date"])
        return frame, mapping


def load_qclass_history_to_engine(
    engine: Any,
    root_or_datafeed: Path | str | QClassVnpyDatafeed,
    *,
    use_ticks: bool = False,
) -> int:
    """Load QClass release history directly into a vn.py BacktestingEngine."""

    datafeed = (
        root_or_datafeed
        if isinstance(root_or_datafeed, QClassVnpyDatafeed)
        else QClassVnpyDatafeed(root_or_datafeed)
    )
    if use_ticks:
        history = datafeed.load_tick_data(
            engine.symbol,
            engine.exchange,
            engine.start,
            engine.end,
        )
    else:
        history = datafeed.load_bar_data(
            engine.symbol,
            engine.exchange,
            engine.interval,
            engine.start,
            engine.end,
        )
    engine.history_data.clear()
    engine.history_data.extend(history)
    patch_engine_history_loader(engine, datafeed)
    return len(history)


def patch_engine_history_loader(engine: Any, datafeed: QClassVnpyDatafeed) -> None:
    """Patch one BacktestingEngine instance so CtaTemplate.load_bar uses QClass data."""

    from vnpy.trader.utility import extract_vt_symbol
    from vnpy_ctastrategy.base import INTERVAL_DELTA_MAP

    def load_bar(
        vt_symbol: str,
        days: int,
        interval: Interval,
        callback: Callable,
        use_database: bool,
    ) -> list[BarData]:
        query_interval = interval
        if interval != Interval.DAILY and getattr(engine, "interval", None) == Interval.DAILY:
            query_interval = Interval.DAILY
        init_end = engine.start - INTERVAL_DELTA_MAP[query_interval]
        init_start = engine.start - timedelta(days=days)
        symbol, exchange = extract_vt_symbol(vt_symbol)
        return datafeed.load_bar_data(
            symbol,
            exchange,
            query_interval,
            init_start,
            init_end,
        )

    def load_tick(
        vt_symbol: str,
        days: int,
        callback: Callable,
    ) -> list[TickData]:
        init_end = engine.start - timedelta(seconds=1)
        init_start = engine.start - timedelta(days=days)
        symbol, exchange = extract_vt_symbol(vt_symbol)
        return datafeed.load_tick_data(symbol, exchange, init_start, init_end)

    engine.load_bar = load_bar
    engine.load_tick = load_tick
    engine.qclass_datafeed = datafeed


DATE_PRIORITY = [
    "publidate",
    "pub_date",
    "anondt",
    "annodt",
    "anndt",
    "trddt",
    "tradingdate",
    "date",
    "idxtrd01",
    "enddate",
    "accper",
    "rptdt",
    "reportdate",
    "report_date",
    "statdt",
    "implementdate",
    "listdt",
]
SYMBOL_PRIORITY = ["stkcd", "stock_id", "stockid", "symbol", "indexcd", "stockcode"]
LOADABLE_SUFFIXES = {".csv": "csv", ".parquet": "parquet"}
CSMAR_TABLE_RE = re.compile(r"^[A-Z]{2,5}[0-9]?_[A-Za-z0-9_]+$")

DAILY_ALIASES: dict[str, tuple[str, ...]] = {
    "open": ("opnprc", "open", "open_price", "idxtrd02"),
    "high": ("hiprc", "high", "high_price", "idxtrd03"),
    "low": ("loprc", "low", "low_price", "idxtrd04"),
    "close": ("clsprc", "close", "close_price", "idxtrd05"),
    "volume": ("dnshrtrd", "volume", "vol", "idxtrd06"),
    "turnover": ("dnvaltrd", "turnover", "amount", "idxtrd07"),
    "pre_close": ("precloseprice", "pre_close", "preclose"),
    "limit_down": ("limitdown", "limit_down", "downlimit"),
    "limit_up": ("limitup", "limit_up", "uplimit"),
}


def scan_catalog(root: Path | str) -> DataCatalog:
    root_path = Path(root)
    datasets: list[DatasetSpec] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root_path).parts):
            continue
        file_format = LOADABLE_SUFFIXES.get(path.suffix.lower())
        if file_format is None:
            continue
        columns = read_columns(path, file_format)
        date_columns = infer_date_columns(columns)
        primary_asof_column = choose_primary_asof_column(date_columns)
        table_code = path.stem if CSMAR_TABLE_RE.match(path.stem) else None
        aliases = [table_code] if table_code else []
        datasets.append(
            DatasetSpec(
                dataset_id=dataset_id_from_path(path.relative_to(root_path)),
                aliases=aliases,
                csmar_table_code=table_code,
                relative_path=path.relative_to(root_path).as_posix(),
                format=file_format,
                is_static=primary_asof_column is None,
                date_columns=date_columns,
                primary_asof_column=primary_asof_column,
                symbol_column=infer_symbol_column(columns),
            )
        )
    return DataCatalog(root=root_path.as_posix(), datasets=datasets)


def read_columns(path: Path, file_format: str) -> list[str]:
    if file_format == "csv":
        return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    if file_format == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("pyarrow is required to inspect parquet datasets") from exc
        return list(pq.read_schema(path).names)
    raise ValueError(f"unsupported dataset format: {file_format}")


def read_dataset(
    path: Path,
    spec: DatasetSpec,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if spec.format == "csv":
        kwargs: dict[str, Any] = {"encoding": "utf-8-sig"}
        if columns is not None:
            kwargs["usecols"] = columns
        if spec.symbol_column and (columns is None or spec.symbol_column in columns):
            kwargs["dtype"] = {spec.symbol_column: str}
        return pd.read_csv(path, **kwargs)
    if spec.format == "parquet":
        return pd.read_parquet(path, columns=columns)
    raise ValueError(f"unsupported dataset format: {spec.format}")


def dataset_id_from_path(relative_path: Path) -> str:
    parts = list(relative_path.with_suffix("").parts)
    if parts and parts[0] in {"distribute", "teacher"}:
        parts = parts[1:]
    return ".".join(parts)


def infer_date_columns(columns: Iterable[str]) -> list[str]:
    known = set(DATE_PRIORITY)
    return [column for column in columns if column.lower() in known]


def choose_primary_asof_column(columns: Iterable[str]) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for name in DATE_PRIORITY:
        if name in by_lower:
            return by_lower[name]
    return None


def infer_symbol_column(columns: Iterable[str]) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for name in SYMBOL_PRIORITY:
        if name in by_lower:
            return by_lower[name]
    return None


def columns_with_required(
    columns: list[str] | None,
    required_columns: list[str],
) -> list[str] | None:
    if columns is None:
        return None
    merged = list(columns)
    for column in required_columns:
        if column is not None and column not in merged:
            merged.append(column)
    return merged


def project_columns(df: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    if columns is None:
        return df.copy()
    return df.loc[:, columns].copy()


def coerce_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: list[str] | None,
) -> pd.DataFrame:
    if not numeric_columns:
        return df
    missing = [column for column in numeric_columns if column not in df.columns]
    if missing:
        raise KeyError(f"numeric columns missing from returned data: {missing}")
    result = df.copy()
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def to_timestamp(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="raise")
    if isinstance(timestamp, pd.DatetimeIndex):
        raise ValueError("expected one date, got many")
    return pd.Timestamp(timestamp).normalize()


def date_series(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m", "%Y/%m", "%Y%m"):
        missing = parsed.isna() & values.notna() & (values != "")
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(
            values.loc[missing],
            format=date_format,
            errors="coerce",
        )
    missing = parsed.isna() & values.notna() & (values != "")
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], errors="coerce")
    return parsed.dt.normalize()


def daily_column_mapping(
    bundle: QClassDataBundle,
    dataset_id_or_alias: str,
) -> dict[str, str]:
    spec = bundle.describe(dataset_id_or_alias)
    columns = read_columns(bundle.path(dataset_id_or_alias), spec.format)
    mapping: dict[str, str] = {}
    mapping["symbol"] = spec.symbol_column or first_existing(columns, SYMBOL_PRIORITY)
    mapping["date"] = spec.primary_asof_column or first_existing(columns, DATE_PRIORITY)
    for field_name, aliases in DAILY_ALIASES.items():
        match = first_existing(columns, aliases, required=False)
        if match is not None:
            mapping[field_name] = match
    missing = [field for field in ("symbol", "date", "close") if not mapping.get(field)]
    if missing:
        raise KeyError(
            f"{dataset_id_or_alias} is missing required daily-bar columns: {missing}"
        )
    return mapping


def first_existing(
    columns: Iterable[str],
    aliases: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in by_lower:
            return by_lower[alias.lower()]
    if required:
        raise KeyError(f"none of these columns are present: {list(aliases)}")
    return None


def normalize_symbol(value: Any) -> str:
    text = str(value).strip().upper()
    if "." in text:
        text = text.split(".", 1)[0]
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return text.zfill(6)
    return text


def infer_exchange(symbol: str) -> Exchange:
    normalized = normalize_symbol(symbol)
    if normalized.startswith(("6", "9")):
        return Exchange.SSE
    if normalized.startswith(("4", "8")):
        return Exchange.BSE
    return Exchange.SZSE


def daily_row_to_bar(
    row: pd.Series,
    mapping: dict[str, str],
    *,
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    gateway_name: str,
) -> BarData:
    close_price = row_float(row, mapping, "close")
    return BarData(
        gateway_name=gateway_name,
        symbol=symbol,
        exchange=exchange,
        datetime=row_datetime(row, mapping),
        interval=interval,
        volume=row_float(row, mapping, "volume", default=0.0),
        turnover=row_float(row, mapping, "turnover", default=0.0),
        open_interest=0.0,
        open_price=row_float(row, mapping, "open", default=close_price),
        high_price=row_float(row, mapping, "high", default=close_price),
        low_price=row_float(row, mapping, "low", default=close_price),
        close_price=close_price,
    )


def daily_row_to_tick(
    row: pd.Series,
    mapping: dict[str, str],
    *,
    symbol: str,
    exchange: Exchange,
    gateway_name: str,
    pricetick: float,
) -> TickData:
    close_price = row_float(row, mapping, "close")
    bid_price = max(0.0, close_price - pricetick)
    ask_price = close_price + pricetick
    return TickData(
        gateway_name=gateway_name,
        symbol=symbol,
        exchange=exchange,
        datetime=row_datetime(row, mapping),
        volume=row_float(row, mapping, "volume", default=0.0),
        turnover=row_float(row, mapping, "turnover", default=0.0),
        last_price=close_price,
        last_volume=row_float(row, mapping, "volume", default=0.0),
        limit_up=row_float(row, mapping, "limit_up", default=0.0),
        limit_down=row_float(row, mapping, "limit_down", default=0.0),
        open_price=row_float(row, mapping, "open", default=close_price),
        high_price=row_float(row, mapping, "high", default=close_price),
        low_price=row_float(row, mapping, "low", default=close_price),
        pre_close=row_float(row, mapping, "pre_close", default=0.0),
        bid_price_1=bid_price,
        ask_price_1=ask_price,
        bid_volume_1=row_float(row, mapping, "volume", default=0.0),
        ask_volume_1=row_float(row, mapping, "volume", default=0.0),
    )


def row_datetime(row: pd.Series, mapping: dict[str, str]) -> datetime:
    timestamp = pd.to_datetime(row[mapping["date"]])
    return datetime.combine(pd.Timestamp(timestamp).date(), time(hour=15))


def row_float(
    row: pd.Series,
    mapping: dict[str, str],
    field_name: str,
    *,
    default: float | None = None,
) -> float:
    column = mapping.get(field_name)
    if column is None:
        if default is None:
            raise KeyError(f"missing required mapped field: {field_name}")
        return default
    value = pd.to_numeric(row[column], errors="coerce")
    if pd.isna(value):
        if default is None:
            raise ValueError(f"cannot convert {column}={row[column]!r} to float")
        return default
    return float(value)


__all__ = [
    "DataCatalog",
    "DatasetSpec",
    "FutureDataAccessError",
    "QClassDataBundle",
    "QClassDatedData",
    "QClassVnpyDatabase",
    "QClassVnpyDatafeed",
    "ReadOnlyDataError",
    "daily_row_to_bar",
    "daily_row_to_tick",
    "infer_exchange",
    "load_qclass_history_to_engine",
    "normalize_symbol",
    "patch_engine_history_loader",
    "scan_catalog",
]
