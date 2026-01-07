from __future__ import annotations

from typing import Optional, Any
from pathlib import Path
import json
import os
import time

import pandas as pd
import numpy as np
import requests


ZONOS_GRAPHQL_URL = "https://api.zonos.com/graphql"


COO_MUTATION = """
mutation CountryOfOriginInfer($input: [CountryOfOriginInferenceInput!]!) {
  countryOfOriginInfer(input: $input) {
    brand
    categories
    confidenceScore
    countryOfOrigin
    name
    description
    material
    alternates {
      countryOfOrigin
      probabilityMass
    }
  }
}
"""


def infer_country_of_origin(
    records: list[dict],
    api_key: Optional[str] = None,
    batch_size: int = 50,
    timeout_s: int = 60,
    retry_attempts: int = 3,
    retry_backoff_s: float = 1.5,
) -> pd.DataFrame:
    """
    Call Zonos countryOfOriginInfer GraphQL mutation for a list of normalized records.
    Returns a DataFrame with original inputs + inferred fields appended.

    IMPORTANT: Omits keys whose values are None from the request payload.
    """
    if api_key is None:
        api_key = os.getenv("ZONOS_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing API key. Provide api_key=... or set environment variable ZONOS_API_KEY."
        )

    # Convert to DataFrame so we can easily attach output columns later
    df_in = pd.DataFrame(records)

    # Prepare request inputs: GraphQL expects camelCase fields
    gql_inputs = [to_graphql_input(r) for r in records]

    results: list[dict] = []

    for start in range(0, len(gql_inputs), batch_size):
        batch = gql_inputs[start : start + batch_size]

        payload = {
            "query": COO_MUTATION,
            "variables": {"input": batch},
        }

        response_data = _post_graphql_with_retries(
            url=ZONOS_GRAPHQL_URL,
            payload=payload,
            api_key=api_key,
            timeout_s=timeout_s,
            retry_attempts=retry_attempts,
            retry_backoff_s=retry_backoff_s,
        )

        # GraphQL shape: {"data": {"countryOfOriginInfer": [...]}}
        batch_out = response_data["data"]["countryOfOriginInfer"]

        if len(batch_out) != len(batch):
            raise RuntimeError(
                f"Unexpected API result length. Sent {len(batch)} inputs but got {len(batch_out)} outputs."
            )

        results.extend(batch_out)

    # Build output frame of inferred values
    df_out = pd.DataFrame(results)

    # Rename + normalize inferred columns
    df_out = df_out.rename(
        columns={
            "confidenceScore": "inferred_confidence_score",
            "countryOfOrigin": "inferred_country_of_origin",
            "brand": "inferred_brand",
            "name": "inferred_name",
            "description": "inferred_description",
            "material": "inferred_material",
            "categories": "inferred_categories",
        }
    )

    # alternates is a list[dict] — store as JSON string so it's file-friendly
    if "alternates" in df_out.columns:
        df_out["inferred_alternates"] = df_out["alternates"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else None
        )
        df_out = df_out.drop(columns=["alternates"])

    # Combine original + inferred columns side-by-side
    # keep only confidence score country of origin and alternates
    df_out = df_out[
        [
            "inferred_confidence_score",
            "inferred_country_of_origin",
            "inferred_alternates",
        ]
    ]
    df_final = pd.concat(
        [df_in.reset_index(drop=True), df_out.reset_index(drop=True)], axis=1
    )

    return df_final


def to_graphql_input(record: dict) -> dict:
    """
    Map normalized record (snake_case) to GraphQL input (camelCase),
    omitting keys whose values are None.
    """
    mapping = {
        "brand": "brand",
        "name": "name",
        "categories": "categories",
        "description": "description",
        "material": "material",
        "ship_from_country": "shipFromCountry",
        "primary_target_country_code": "primaryTargetCountryCode",
        "amount": "amount",
        "currency_code": "currencyCode",
    }

    out: dict[str, Any] = {}

    for k_in, k_out in mapping.items():
        if k_in not in record:
            continue
        v = record[k_in]
        if v is None:
            continue

        # If categories is None or empty, omit it too
        if k_in == "categories":
            if not isinstance(v, list) or len(v) == 0:
                continue

        out[k_out] = v

    return out


def _post_graphql_with_retries(
    url: str,
    payload: dict,
    api_key: str,
    timeout_s: int,
    retry_attempts: int,
    retry_backoff_s: float,
) -> dict:
    headers = {
        "Content-Type": "application/json",
        # Adjust this if your auth scheme differs (some setups want "Authorization: Bearer <token>")
        "credentialToken": api_key,
    }

    last_err = None

    for attempt in range(1, retry_attempts + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
            resp.raise_for_status()

            data = resp.json()

            # GraphQL-level errors
            if "errors" in data and data["errors"]:
                raise RuntimeError(f"GraphQL errors: {data['errors']}")

            if "data" not in data:
                raise RuntimeError(f"Unexpected response shape: {data}")

            return data

        except Exception as e:
            last_err = e
            if attempt < retry_attempts:
                sleep_for = retry_backoff_s * attempt
                time.sleep(sleep_for)
            else:
                raise RuntimeError(
                    f"GraphQL request failed after {retry_attempts} attempts: {e}"
                ) from e

    # unreachable
    raise RuntimeError(f"GraphQL request failed: {last_err}")


def infer_from_file(
    file_path: str,
    name_columns: Optional[list[str]] = ["name"],
    brand_columns: Optional[list[str]] = ["brand"],
    categories_columns: Optional[list[str]] = ["categories"],
    description_columns: Optional[list[str]] = ["description"],
    material_columns: Optional[list[str]] = ["material"],
    ship_from_country_column: Optional[str] = "ship_from_country",
    primary_target_country_code_column: Optional[str] = "primary_target_country_code",
    amount_column: Optional[str] = "amount",
    currency_code_column: Optional[str] = "currency_code",
) -> list[dict]:
    """
    Read a file into a pandas DataFrame and build a normalized DataFrame with columns:
      - name (required)
      - brand
      - categories (list[str])
      - description
      - material
      - ship_from_country
      - primary_target_country_code
      - amount
      - currency_code

    Rules:
      - If source columns passed as parameters are missing, treat them as all nulls.
      - For non-name columns: empty strings / whitespace-only strings => null (None).
      - name is required: cannot be null/empty/whitespace-only (raises ValueError).
      - brand/description/material: if multiple source columns provided, concatenate values with ' | '
      - categories: if multiple source columns provided, build a list[str] containing all non-null values.
    """
    src = _read_file_to_df(file_path)

    # Normalize parameter defaults (avoid None)
    name_columns = name_columns or ["name"]
    brand_columns = brand_columns or ["brand"]
    categories_columns = categories_columns or ["categories"]
    description_columns = description_columns or ["description"]
    material_columns = material_columns or ["material"]

    # Build normalized output columns
    out = pd.DataFrame(index=src.index)

    # NAME (required)
    out["name"] = _combine_text_columns(
        src=src,
        cols=name_columns,
        sep=" | ",
        missing_as_null=True,
        strip=True,
        empty_as_null=False,  # for name we don't allow empty => handled below
    )

    # Validate name
    name_series = out["name"]
    # Treat whitespace-only as invalid too
    invalid_mask = name_series.isna() | (name_series.astype(str).str.strip() == "")
    if invalid_mask.any():
        bad_rows = src.index[invalid_mask].tolist()
        raise ValueError(
            f"`name` is required and cannot be null/empty. "
            f"Invalid name found on rows: {bad_rows[:50]}"
            + (" ..." if len(bad_rows) > 50 else "")
        )

    # BRAND (concat)
    out["brand"] = _combine_text_columns(
        src=src,
        cols=brand_columns,
        sep=" | ",
        missing_as_null=True,
        strip=True,
        empty_as_null=True,
    )

    # DESCRIPTION (concat)
    out["description"] = _combine_text_columns(
        src=src,
        cols=description_columns,
        sep=" | ",
        missing_as_null=True,
        strip=True,
        empty_as_null=True,
    )

    # MATERIAL (concat)
    out["material"] = _combine_text_columns(
        src=src,
        cols=material_columns,
        sep=" | ",
        missing_as_null=True,
        strip=True,
        empty_as_null=True,
    )

    # CATEGORIES (list[str])
    out["categories"] = _combine_list_columns(
        src=src,
        cols=categories_columns,
        missing_as_null=True,
        strip=True,
        empty_as_null=True,
    )

    # Single-column passthrough fields
    out["ship_from_country"] = _single_column_or_null(
        src=src,
        col=ship_from_country_column,
        strip=True,
        empty_as_null=True,
    )

    out["primary_target_country_code"] = _single_column_or_null(
        src=src,
        col=primary_target_country_code_column,
        strip=True,
        empty_as_null=True,
    )

    out["amount"] = _single_column_or_null(
        src=src,
        col=amount_column,
        strip=False,
        empty_as_null=True,
    )

    out["currency_code"] = _single_column_or_null(
        src=src,
        col=currency_code_column,
        strip=True,
        empty_as_null=True,
    )

    # Replace pandas NaN with None for clean dict output
    out = out.replace({np.nan: None})

    return out.to_dict(orient="records")


def _read_file_to_df(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)

    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported: "
        ".csv, .tsv, .xlsx, .parquet, .jsonl, .json"
    )


def _get_column_or_null(src: pd.DataFrame, col: str) -> pd.Series:
    """Return src[col] if exists, else a series of nulls."""
    if col in src.columns:
        return src[col]
    return pd.Series([None] * len(src), index=src.index)


def _clean_series(
    s: pd.Series,
    *,
    strip: bool = True,
    empty_as_null: bool = True,
) -> pd.Series:
    """Clean a series: convert whitespace-only to None (if empty_as_null)."""
    # Keep original nulls
    s = s.copy()

    # Convert non-null to string only when needed
    # But preserve numbers, etc. for amount column
    if strip or empty_as_null:
        # Only operate on object/string-like values; others leave as-is
        s_obj = s.astype("object")

        def _clean_val(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            if isinstance(v, str):
                vv = v.strip() if strip else v
                if empty_as_null and vv.strip() == "":
                    return None
                return vv
            # For non-strings, just return as-is
            return v

        return s_obj.map(_clean_val)

    return s


def _combine_text_columns(
    *,
    src: pd.DataFrame,
    cols: list[str],
    sep: str = " | ",
    missing_as_null: bool = True,
    strip: bool = True,
    empty_as_null: bool = True,
) -> pd.Series:
    """
    Combine multiple columns into one text column.
    - Missing columns treated as nulls if missing_as_null True.
    - Clean values (strip, empty->null).
    - Join non-null parts with sep.
    - If all parts null => null.
    """
    if not cols:
        return pd.Series([None] * len(src), index=src.index)

    series_list = []
    for c in cols:
        s = _get_column_or_null(src, c) if missing_as_null else src[c]
        s = _clean_series(s, strip=strip, empty_as_null=empty_as_null)
        series_list.append(s)

    def _join_row(values):
        parts = []
        for v in values:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            # Convert non-strings to string for join behavior
            parts.append(str(v))
        if not parts:
            return None
        return sep.join(parts)

    combined = pd.concat(series_list, axis=1).apply(
        lambda row: _join_row(row.values), axis=1
    )
    return combined


def _combine_list_columns(
    *,
    src: pd.DataFrame,
    cols: list[str],
    missing_as_null: bool = True,
    strip: bool = True,
    empty_as_null: bool = True,
) -> pd.Series:
    """
    Combine multiple columns into a list[str]:
    - Each row becomes a list of all non-null, cleaned values from the specified cols.
    - Empty/whitespace values become null if empty_as_null.
    - If list is empty => null (None).
    """
    if not cols:
        return pd.Series([None] * len(src), index=src.index)

    series_list = []
    for c in cols:
        s = _get_column_or_null(src, c) if missing_as_null else src[c]
        s = _clean_series(s, strip=strip, empty_as_null=empty_as_null)
        series_list.append(s)

    def _list_row(values):
        out = []
        for v in values:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            out.append(str(v))
        return out if out else None

    combined = pd.concat(series_list, axis=1).apply(
        lambda row: _list_row(row.values), axis=1
    )
    return combined


def _single_column_or_null(
    *,
    src: pd.DataFrame,
    col: Optional[str],
    strip: bool = True,
    empty_as_null: bool = True,
) -> pd.Series:
    """Return cleaned column if present, else null series."""
    if not col:
        return pd.Series([None] * len(src), index=src.index)

    s = _get_column_or_null(src, col)
    return _clean_series(s, strip=strip, empty_as_null=empty_as_null)


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    file_path = "/home/lancecondie/Documents/repos/zns-coo-inference/SAC_test_missing_hs_coo.xlsx"

    results = infer_country_of_origin(
        records=infer_from_file(
            file_path,
            name_columns=["PROCECO_PN"],
            description_columns=[
                "PN_DESCRIPTION",
                "EXTENDED_DESCRIPTION",
                "ENGLISH_DESCRIPTION",
            ],
        ),
        api_key=os.getenv("ZONOS_API_KEY"),
    )

    results.to_csv(f"{file_path}_inferred.csv")
