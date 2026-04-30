from __future__ import annotations

import time
from pathlib import Path

import pyarrow as pa
from pyarrow import csv as pacsv
from pyarrow import parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "raw" / "pq"

BASE_SCHEMA: dict[str, pa.DataType] = {
    "srch_id": pa.uint32(),
    "date_time": pa.timestamp("s"),
    "site_id": pa.uint16(),
    "visitor_location_country_id": pa.uint16(),
    "visitor_hist_starrating": pa.float32(),
    "visitor_hist_adr_usd": pa.float32(),
    "prop_country_id": pa.uint16(),
    "prop_id": pa.uint32(),
    "prop_starrating": pa.uint8(),
    "prop_review_score": pa.float32(),
    "prop_brand_bool": pa.uint8(),
    "prop_location_score1": pa.float32(),
    "prop_location_score2": pa.float32(),
    "prop_log_historical_price": pa.float32(),
    "price_usd": pa.float32(),
    "promotion_flag": pa.uint8(),
    "srch_destination_id": pa.uint32(),
    "srch_length_of_stay": pa.uint8(),
    "srch_booking_window": pa.uint16(),
    "srch_adults_count": pa.uint8(),
    "srch_children_count": pa.uint8(),
    "srch_room_count": pa.uint8(),
    "srch_saturday_night_bool": pa.uint8(),
    "srch_query_affinity_score": pa.float32(),
    "orig_destination_distance": pa.float32(),
    "random_bool": pa.uint8(),
}
for i in range(1, 9):
    BASE_SCHEMA[f"comp{i}_rate"] = pa.int8()
    BASE_SCHEMA[f"comp{i}_inv"] = pa.int8()
    BASE_SCHEMA[f"comp{i}_rate_percent_diff"] = pa.float32()

TRAIN_EXTRA: dict[str, pa.DataType] = {
    "position": pa.uint8(),
    "click_bool": pa.uint8(),
    "gross_bookings_usd": pa.float32(),
    "booking_bool": pa.uint8(),
}


def convert(src: Path, dst: Path, extra_cols: dict[str, pa.DataType] | None = None) -> None:
    cols = {**BASE_SCHEMA, **(extra_cols or {})}

    convert_options = pacsv.ConvertOptions(
        column_types=cols,
        null_values=["NULL", ""],
        strings_can_be_null=True,
    )
    read_options = pacsv.ReadOptions(block_size=64 << 20)

    t0 = time.perf_counter()
    print(f"[{src.name}] streaming -> {dst.name}")
    with pacsv.open_csv(src, read_options=read_options, convert_options=convert_options) as reader:
        writer: pq.ParquetWriter | None = None
        rows = 0
        for batch in reader:
            if writer is None:
                writer = pq.ParquetWriter(dst, batch.schema, compression="snappy")
            writer.write_batch(batch)
            rows += batch.num_rows
            print(f"  {rows:>10,} rows", end="\r")
        if writer is not None:
            writer.close()
    elapsed = time.perf_counter() - t0
    size_mb = dst.stat().st_size / (1 << 20)
    print(f"  {rows:>10,} rows  |  {size_mb:,.1f} MB  |  {elapsed:.1f}s")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    convert(RAW / "training_set_VU_DM.csv", OUT / "train.parquet", TRAIN_EXTRA)
    convert(RAW / "test_set_VU_DM.csv", OUT / "test.parquet")


if __name__ == "__main__":
    main()
