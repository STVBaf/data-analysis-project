# -*- coding: utf-8 -*-
"""
clean_data.py
=============
将两份原始数据清洗为规范的中间文件：

输入：
  data/job_data.csv     (GB18030, 297城 x 2000-2024, 含真实平均工资/GDP/人口/就业)
  data/rent_data.xlsx   (宽表, 71城为列, 2018-11~2025-09 月度, 元/平方米/月)

输出（data/clean/）：
  job_panel.csv         15城 x 年份 面板：真实工资(年/月)、GDP、人口、就业、失业
  rent_panel.csv        15城 x 月份 面板：商品房租金(元/㎡/月) + 估算整租月租金
  city_snapshot.csv     15城最新年份快照，供 YCSI 主流程直接接入

注：
  - 工资口径为"城镇非私营单位在岗职工平均工资"(年薪，元)，月薪 = 年薪/12。
  - 租金原始单位为元/㎡/月；整租月租金 = 单价 × 假设套均面积(AREA_SQM)。
"""
from __future__ import annotations

import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CLEAN = DATA / "clean"
CLEAN.mkdir(exist_ok=True)

TARGET_CITIES = [
    "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "成都",
    "重庆", "武汉", "西安", "长沙", "郑州", "天津", "青岛",
]

# 估算整租月租金所用的套均建筑面积（㎡）。毕业生整租一居室，取 50。
AREA_SQM = 50

# 快照基准年。job_data 平均工资字段在 2021 年后断档，2021 为最新完整年份。
BASE_YEAR = 2021

JOB_RAW_COLS = {
    "年份": "year",
    "城市": "city_raw",
    "地区生产总值(万元)": "gdp_wan",
    "人均地区生产总值(元)": "gdp_per_capita",
    "户籍人口(万人)": "population_wan",
    "城镇非私营单位从业人员数(万人)": "urban_employment_wan",
    "年末城镇登记失业人员数(人)": "registered_unemployed",
    "城镇非私营单位在岗职工平均工资(元)": "avg_wage_year",
}


def read_csv_fallback(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法识别编码: {path}")


def clean_job() -> pd.DataFrame:
    """清洗 job_data.csv → 15城面板。"""
    raw = read_csv_fallback(DATA / "job_data.csv")
    raw = raw.rename(columns=JOB_RAW_COLS)

    # 城市名去掉"市"后缀以匹配目标列表
    raw["city"] = raw["city_raw"].astype(str).str.replace("市", "", regex=False)
    panel = raw[raw["city"].isin(TARGET_CITIES)].copy()

    panel["year"] = panel["year"].astype(int)
    # 数值化
    num_cols = [
        "gdp_wan", "gdp_per_capita", "population_wan",
        "urban_employment_wan", "registered_unemployed", "avg_wage_year",
    ]
    for col in num_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    # 衍生字段
    panel["avg_salary_month"] = (panel["avg_wage_year"] / 12).round(0)
    panel["gdp_yi"] = (panel["gdp_wan"] / 10000).round(2)  # 万元→亿元

    keep = [
        "year", "city", "avg_wage_year", "avg_salary_month",
        "gdp_yi", "gdp_per_capita", "population_wan",
        "urban_employment_wan", "registered_unemployed",
    ]
    panel = panel[keep].sort_values(["city", "year"]).reset_index(drop=True)
    panel.to_csv(CLEAN / "job_panel.csv", index=False, encoding="utf-8-sig")
    return panel


def clean_rent() -> pd.DataFrame:
    """清洗 rent_data.xlsx 宽表 → 15城月度长表面板。"""
    raw = pd.read_excel(DATA / "rent_data.xlsx", sheet_name=0)
    raw = raw.drop(columns=[c for c in raw.columns if str(c).startswith("Unnamed")])

    # 第0,1行是"频率/单位"元数据，第2行起才是日期数据
    date_col = raw.columns[0]
    data = raw.iloc[2:].copy()
    data = data.rename(columns={date_col: "date"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])

    # 列名 "商品房平均出租价格:上海" → "上海"
    rename_map = {}
    for col in data.columns:
        if col == "date":
            continue
        city = str(col).split(":")[-1]
        rename_map[col] = city
    data = data.rename(columns=rename_map)

    # 只保留目标城市中存在的列
    available = [c for c in TARGET_CITIES if c in data.columns]
    missing = [c for c in TARGET_CITIES if c not in data.columns]
    if missing:
        print(f"[rent] 警告：以下目标城市在租金表中缺失: {missing}")

    long = data.melt(
        id_vars=["date"], value_vars=available,
        var_name="city", value_name="rent_per_sqm",
    )
    long["rent_per_sqm"] = pd.to_numeric(long["rent_per_sqm"], errors="coerce")
    long = long.dropna(subset=["rent_per_sqm"])
    long["year"] = long["date"].dt.year
    long["month"] = long["date"].dt.month
    long["rent_month_est"] = (long["rent_per_sqm"] * AREA_SQM).round(0)

    long = long[["date", "year", "month", "city", "rent_per_sqm", "rent_month_est"]]
    long = long.sort_values(["city", "date"]).reset_index(drop=True)
    long.to_csv(CLEAN / "rent_panel.csv", index=False, encoding="utf-8-sig")
    return long


def build_snapshot(job: pd.DataFrame, rent: pd.DataFrame) -> pd.DataFrame:
    """构建 15 城基准年(BASE_YEAR)快照，工资与租金严格同年。"""
    rows = []
    for city in TARGET_CITIES:
        j = job[(job["city"] == city) & (job["year"] == BASE_YEAR)]
        j = j.dropna(subset=["avg_wage_year"])
        if j.empty:
            print(f"[snapshot] {city} 在 {BASE_YEAR} 年无有效工资数据，跳过")
            continue
        jr = j.iloc[0]
        ref_year = BASE_YEAR

        # 租金：取基准年的年均单价；若该年无数据，回退到最近12个月均值
        r_city = rent[rent["city"] == city]
        r_same = r_city[r_city["year"] == ref_year]
        if not r_same.empty:
            rent_sqm = r_same["rent_per_sqm"].mean()
            rent_year_used = ref_year
        elif not r_city.empty:
            latest = r_city.sort_values("date").tail(12)
            rent_sqm = latest["rent_per_sqm"].mean()
            rent_year_used = int(latest["year"].iloc[-1])
        else:
            rent_sqm = np.nan
            rent_year_used = None

        rent_month = round(rent_sqm * AREA_SQM, 0) if pd.notna(rent_sqm) else np.nan
        salary_month = jr["avg_salary_month"]
        ratio = round(rent_month / salary_month, 4) if pd.notna(rent_month) else np.nan

        rows.append({
            "city": city,
            "wage_year_ref": ref_year,
            "avg_wage_year": jr["avg_wage_year"],
            "avg_salary_month": salary_month,
            "gdp_yi": jr["gdp_yi"],
            "gdp_per_capita": jr["gdp_per_capita"],
            "population_wan": jr["population_wan"],
            "urban_employment_wan": jr["urban_employment_wan"],
            "rent_year_ref": rent_year_used,
            "rent_per_sqm": round(rent_sqm, 2) if pd.notna(rent_sqm) else np.nan,
            "rent_month_est": rent_month,
            "rent_income_ratio": ratio,
        })
    snap = pd.DataFrame(rows)
    snap.to_csv(CLEAN / "city_snapshot.csv", index=False, encoding="utf-8-sig")
    return snap


def main() -> None:
    print("=" * 64)
    print("STEP 1  清洗 job_data.csv")
    job = clean_job()
    print(f"  job_panel: {job.shape}, 城市数={job['city'].nunique()}, "
          f"年份={job['year'].min()}-{job['year'].max()}")

    print("STEP 2  清洗 rent_data.xlsx")
    rent = clean_rent()
    print(f"  rent_panel: {rent.shape}, 城市数={rent['city'].nunique()}, "
          f"月份={rent['date'].min():%Y-%m}~{rent['date'].max():%Y-%m}")

    print("STEP 3  构建 15 城快照")
    snap = build_snapshot(job, rent)
    print(f"  city_snapshot: {snap.shape}")
    print("\n--- 15 城快照（真实工资 + 真实租金 + 租金收入比）---")
    show = snap[[
        "city", "wage_year_ref", "avg_salary_month",
        "rent_year_ref", "rent_per_sqm", "rent_month_est", "rent_income_ratio",
    ]]
    print(show.to_string(index=False))
    print(f"\n输出目录: {CLEAN}")


if __name__ == "__main__":
    main()
