from __future__ import annotations

import os
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans


warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 真实数据基准年（与 clean_data.py 的 BASE_YEAR 一致）。
BASE_YEAR = 2021

CITY_COORDS = {
    "北京": (116.4074, 39.9042),
    "上海": (121.4737, 31.2304),
    "广州": (113.2644, 23.1291),
    "深圳": (114.0579, 22.5431),
    "杭州": (120.1551, 30.2741),
    "南京": (118.7969, 32.0603),
    "苏州": (120.5853, 31.2989),
    "成都": (104.0665, 30.5728),
    "重庆": (106.5516, 29.5630),
    "武汉": (114.3054, 30.5931),
    "西安": (108.9398, 34.3416),
    "长沙": (112.9388, 28.2282),
    "郑州": (113.6254, 34.7466),
    "天津": (117.2000, 39.1333),
    "青岛": (120.3826, 36.0671),
}

COORDS_BY_ORDER = [
    (116.4074, 39.9042),
    (121.4737, 31.2304),
    (113.2644, 23.1291),
    (114.0579, 22.5431),
    (120.1551, 30.2741),
    (118.7969, 32.0603),
    (120.5853, 31.2989),
    (104.0665, 30.5728),
    (106.5516, 29.5630),
    (114.3054, 30.5931),
    (108.9398, 34.3416),
    (112.9388, 28.2282),
    (113.6254, 34.7466),
    (117.2000, 39.1333),
    (120.3826, 36.0671),
]

POI_COLS = ["subway", "hospital", "park", "mall", "restaurant", "library", "gym"]
YCSI_FEATURES = [
    "opportunity_score",
    "rent_pressure",
    "life_score",
    "growth_score",
    "commute_score",
]
WEIGHTS = {
    "opportunity": 0.30,
    "life": 0.20,
    "growth": 0.20,
    "rent": 0.20,
    "commute": 0.10,
}

# YCSI 展示分区间：避免首尾城市被钉死为 0/100，体现"相对指数"含义。
SCORE_FLOOR = 40.0
SCORE_CEIL = 95.0

CHART_PLAN = [
    {
        "chart": "全国城市生存力指数点状热力图",
        "question": "15 个目标城市的综合友好度与空间分布",
        "fields": "city, lng, lat, YCSI, population",
        "output": "02_ycsi_city_heat_points.png",
    },
    {
        "chart": "YCSI 排名柱状图",
        "question": "哪些城市更适合作为毕业后的第一座城市",
        "fields": "YCSI, opportunity, life, growth, rent pressure, commute pressure",
        "output": "01_ycsi_ranking.png",
    },
    {
        "chart": "工资 vs 租金收入比气泡图",
        "question": "就业机会与租房压力是否匹配",
        "fields": "avg_salary, avg_rent, rent_income_ratio, job_count",
        "output": "03_salary_rent_bubble.png",
    },
    {
        "chart": "机会-居住压力四象限图",
        "question": "城市属于高机会高压力、低压稳健等哪类",
        "fields": "opportunity_score, rent_pressure, YCSI",
        "output": "04_opportunity_pressure_quadrant.png",
    },
    {
        "chart": "城市指标雷达图",
        "question": "典型城市在机会、居住、通勤、生活、成长上的差异",
        "fields": "five YCSI dimensions",
        "output": "05_city_radar.png",
    },
    {
        "chart": "生活便利 POI 构成图",
        "question": "城市日常生活便利度差异",
        "fields": "subway, hospital, park, mall, restaurant, library, gym",
        "output": "06_poi_convenience_stack.png",
    },
    {
        "chart": "指标相关性热力图",
        "question": "薪资、租金、生活便利、就业机会之间的关系",
        "fields": "salary, rent, job count, YCSI dimensions",
        "output": "07_metric_correlation_heatmap.png",
    },
    {
        "chart": "K-Means 城市聚类图",
        "question": "推荐不同类型青年适合的城市类别",
        "fields": "five YCSI dimensions",
        "output": "08_city_clusters.png",
    },
    {
        "chart": "商品房租金月度趋势图",
        "question": "各城市租金随时间的变化（2018-2025）",
        "fields": "date, city, rent_per_sqm",
        "output": "09_rent_trend.png",
    },
]


def read_csv_fallback(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median())
    span = values.max() - values.min()
    if pd.isna(span) or span == 0:
        return pd.Series(np.full(len(values), 0.5), index=series.index)
    return (values - values.min()) / span


def save_figure(filename: str) -> Path:
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def write_chart_plan() -> Path:
    lines = [
        "# YCSI 可视化图表清单",
        "",
        "| 图表 | 对应研究问题 | 主要字段 | 输出文件 |",
        "|---|---|---|---|",
    ]
    for item in CHART_PLAN:
        lines.append(
            f"| {item['chart']} | {item['question']} | {item['fields']} | {item['output']} |"
        )
    lines.extend(
        [
            "",
            "可选扩展：通勤-租房地图需要 `rent_data.csv` 中的区县/房源级租金、经纬度和就业中心数据；文本词云需要 `city_text.csv` 的 `city,text` 评论数据。",
            "",
        ]
    )
    path = OUTPUT_DIR / "chart_plan.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def load_base_data() -> pd.DataFrame:
    city_basic = read_csv_fallback(DATA_DIR / "city_basic.csv")
    poi = read_csv_fallback(DATA_DIR / "poi_data.csv")

    industry_path = PROJECT_ROOT / "city_basic_15_with_industry.csv"
    if industry_path.exists():
        industry = read_csv_fallback(industry_path)[
            [
                "city",
                "primary_industry_pct",
                "secondary_industry_pct",
                "tertiary_industry_pct",
            ]
        ]
        city_basic = city_basic.merge(industry, on="city", how="left")
    else:
        city_basic["primary_industry_pct"] = np.nan
        city_basic["secondary_industry_pct"] = np.nan
        city_basic["tertiary_industry_pct"] = np.nan

    for frame in (city_basic, poi):
        frame.columns = [str(column).strip() for column in frame.columns]

    metrics = city_basic.merge(poi, on="city", how="left")
    numeric_cols = [
        "gdp",
        "population",
        "disposable_income",
        "university_count",
        "metro_lines",
        "primary_industry_pct",
        "secondary_industry_pct",
        "tertiary_industry_pct",
        *POI_COLS,
    ]
    for column in numeric_cols:
        if column in metrics.columns:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
            metrics[column] = metrics[column].fillna(metrics[column].median())

    coords = metrics["city"].map(CITY_COORDS)
    metrics["lng"] = coords.map(lambda item: item[0] if isinstance(item, tuple) else np.nan)
    metrics["lat"] = coords.map(lambda item: item[1] if isinstance(item, tuple) else np.nan)

    if metrics["lng"].isna().any() and len(metrics) == len(COORDS_BY_ORDER):
        ordered = pd.DataFrame(COORDS_BY_ORDER, columns=["lng_fallback", "lat_fallback"])
        metrics["lng"] = metrics["lng"].fillna(ordered["lng_fallback"])
        metrics["lat"] = metrics["lat"].fillna(ordered["lat_fallback"])

    return metrics


def add_job_and_rent_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """接入真实薪资/租金/就业数据（data/clean/city_snapshot.csv）。

    数据来源：
      - 薪资 avg_salary：城镇非私营单位在岗职工平均工资 / 12（2021，元/月）
      - 租金 avg_rent：商品房平均出租价格 × 50㎡（2021，元/月）
      - 岗位规模 job_count：城镇非私营单位从业人员数（万人，真实就业规模代理）
    若快照缺失则回退到宏观指标估算（保留兼容）。
    """
    metrics = metrics.copy()
    source_notes: list[str] = []

    snap_path = CLEAN_DIR / "city_snapshot.csv"
    if snap_path.exists():
        snap = read_csv_fallback(snap_path)
        snap.columns = [str(c).strip() for c in snap.columns]
        cols = {
            "city": "city",
            "avg_salary_month": "avg_salary",
            "rent_month_est": "avg_rent",
            "rent_income_ratio": "rent_income_ratio",
            "urban_employment_wan": "urban_employment_wan",
        }
        snap = snap[list(cols.keys())].rename(columns=cols)
        for col in ("avg_salary", "avg_rent", "rent_income_ratio", "urban_employment_wan"):
            snap[col] = pd.to_numeric(snap[col], errors="coerce")
        # 就业人数（万人）作为真实"岗位规模"代理，替代原模拟 job_count
        snap["job_count"] = (snap["urban_employment_wan"] * 10000).round(0).astype("Int64")
        snap = snap.drop(columns=["urban_employment_wan"])

        metrics = metrics.merge(snap, on="city", how="left")
        source_notes.append(
            f"薪资数据：城镇非私营单位在岗职工平均工资/12（{BASE_YEAR}，真实，data/job_data.csv）。"
        )
        source_notes.append(
            f"租金数据：商品房平均出租价格×50㎡（{BASE_YEAR}，真实，data/rent_data.xlsx）。"
        )
        source_notes.append(
            "岗位规模：以城镇非私营单位从业人员数代理（真实就业规模）。"
        )
    else:
        source_notes.append("未发现 city_snapshot.csv，请先运行 clean_data.py 生成清洗数据。")
        raise FileNotFoundError(f"缺少清洗数据: {snap_path}，请先运行 python clean_data.py")

    # 兜底：极个别缺失用中位数填充，保证后续计算不崩
    for col in ("avg_salary", "avg_rent", "rent_income_ratio"):
        if metrics[col].isna().any():
            metrics[col] = metrics[col].fillna(metrics[col].median())
            source_notes.append(f"提示：{col} 存在缺失，已用中位数填充。")
    metrics["job_count"] = (
        metrics["job_count"].fillna(metrics["job_count"].median()).astype(int)
    )

    return metrics, source_notes


def calculate_ycsi(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = metrics.copy()

    metrics["opportunity_score"] = (
        0.45 * minmax(metrics["avg_salary"])
        + 0.40 * minmax(metrics["job_count"])
        + 0.15 * minmax(metrics["gdp"])
    )

    metrics["rent_pressure"] = minmax(metrics["rent_income_ratio"])

    # 生活便利度按 POI 绝对总量衡量（可达的选择多寡），而非人均。
    # 人均会惩罚大城市规模优势（如北京餐饮5591家被巨大人口稀释为人均垫底），
    # 与"城市能提供多少便利"的直觉相悖。总量同时避免了人口口径问题。
    poi_scaled = metrics[POI_COLS].apply(minmax)
    poi_weights = pd.Series(
        {
            "subway": 0.18,
            "hospital": 0.16,
            "park": 0.14,
            "mall": 0.14,
            "restaurant": 0.16,
            "library": 0.10,
            "gym": 0.12,
        }
    )
    metrics["life_score"] = minmax((poi_scaled * poi_weights).sum(axis=1))

    metrics["gdp_per_capita"] = metrics["gdp"] * 10000 / metrics["population"].replace(0, np.nan)
    metrics["growth_score"] = (
        0.30 * minmax(metrics["gdp_per_capita"])
        + 0.30 * minmax(metrics["disposable_income"])
        + 0.25 * minmax(metrics["university_count"])
        + 0.15 * minmax(metrics["tertiary_industry_pct"])
    )

    # 通勤压力以城区就业规模为主，而非户籍人口（避免郊县人口虚高拉满压力）。
    commute_raw = (
        0.42 * minmax(metrics["job_count"])
        + 0.24 * minmax(metrics["gdp"])
        + 0.14 * minmax(metrics["avg_rent"])
        - 0.20 * minmax(metrics["metro_lines"])
    )
    metrics["commute_score"] = minmax(commute_raw)

    metrics["ycsi_raw"] = (
        WEIGHTS["opportunity"] * metrics["opportunity_score"]
        + WEIGHTS["life"] * metrics["life_score"]
        + WEIGHTS["growth"] * metrics["growth_score"]
        - WEIGHTS["rent"] * metrics["rent_pressure"]
        - WEIGHTS["commute"] * metrics["commute_score"]
    )
    # 线性映射到 [SCORE_FLOOR, SCORE_CEIL]，避免 min-max 把首尾城市钉死为 0/100。
    # YCSI 是相对指数：最低城市仍有基础分，最高城市不取满分。
    raw = metrics["ycsi_raw"]
    span = raw.max() - raw.min()
    if pd.isna(span) or span == 0:
        scaled = pd.Series(np.full(len(raw), (SCORE_FLOOR + SCORE_CEIL) / 2), index=raw.index)
    else:
        scaled = SCORE_FLOOR + (raw - raw.min()) / span * (SCORE_CEIL - SCORE_FLOOR)
    metrics["ycsi"] = scaled.round(2)
    metrics["rank"] = metrics["ycsi"].rank(ascending=False, method="first").astype(int)
    metrics["rent_friendliness"] = 1 - metrics["rent_pressure"]
    metrics["commute_friendliness"] = 1 - metrics["commute_score"]

    return metrics, poi_scaled


def plot_ycsi_ranking(metrics: pd.DataFrame) -> Path:
    plot_df = metrics.sort_values("ycsi", ascending=True)
    colors = plt.cm.viridis(minmax(plot_df["ycsi"]))
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.barh(plot_df["city"], plot_df["ycsi"], color=colors, edgecolor="white", linewidth=0.8)
    for _, row in plot_df.iterrows():
        ax.text(row["ycsi"] + 1.0, row["city"], f"{row['ycsi']:.1f}", va="center", fontsize=9)
    ax.set_xlim(0, SCORE_CEIL + 10)
    ax.set_xlabel(f"YCSI 综合得分（相对指数，{SCORE_FLOOR:.0f}-{SCORE_CEIL:.0f}）")
    ax.set_title("15 城青年城市生存力指数排名")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return save_figure("01_ycsi_ranking.png")


def plot_city_heat_points(metrics: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))
    size = 120 + minmax(metrics["population"]) * 620
    scatter = ax.scatter(
        metrics["lng"],
        metrics["lat"],
        c=metrics["ycsi"],
        s=size,
        cmap="YlOrRd",
        edgecolor="#333333",
        linewidth=0.7,
        alpha=0.88,
    )
    for _, row in metrics.iterrows():
        ax.text(row["lng"] + 0.18, row["lat"] + 0.12, row["city"], fontsize=8.5)
    ax.set_xlim(102, 123)
    ax.set_ylim(21, 41)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_title("15 城 YCSI 空间分布（点状热力图）")
    ax.grid(alpha=0.22)
    color_bar = plt.colorbar(scatter, ax=ax, shrink=0.82)
    color_bar.set_label("YCSI 得分")
    return save_figure("02_ycsi_city_heat_points.png")


def plot_salary_rent_bubble(metrics: pd.DataFrame) -> Path:
    salary_median = metrics["avg_salary"].median()
    ratio_median = metrics["rent_income_ratio"].median()

    def classify(row: pd.Series) -> str:
        high_salary = row["avg_salary"] >= salary_median
        high_pressure = row["rent_income_ratio"] >= ratio_median
        if high_salary and high_pressure:
            return "高薪高压"
        if high_salary and not high_pressure:
            return "高薪宜居"
        if not high_salary and high_pressure:
            return "低薪高压"
        return "低压稳健"

    plot_df = metrics.copy()
    plot_df["salary_rent_quadrant"] = plot_df.apply(classify, axis=1)
    palette = {
        "高薪高压": "#d73027",
        "高薪宜居": "#1a9850",
        "低薪高压": "#fdae61",
        "低压稳健": "#4575b4",
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    for name, group in plot_df.groupby("salary_rent_quadrant"):
        ax.scatter(
            group["avg_salary"],
            group["rent_income_ratio"],
            s=120 + minmax(group["job_count"]) * 900,
            color=palette[name],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.8,
            label=name,
        )
        for _, row in group.iterrows():
            ax.text(row["avg_salary"] + 55, row["rent_income_ratio"] + 0.003, row["city"], fontsize=8.5)
    ax.axvline(salary_median, color="#555555", linestyle="--", linewidth=1)
    ax.axhline(ratio_median, color="#555555", linestyle="--", linewidth=1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("平均月薪")
    ax.set_ylabel("租金收入比")
    ax.set_title("工资 vs 租金收入比气泡图（气泡大小=岗位数量）")
    ax.grid(alpha=0.22)
    ax.legend(title="城市类型", frameon=False, loc="best")
    return save_figure("03_salary_rent_bubble.png")


def plot_opportunity_pressure_quadrant(metrics: pd.DataFrame) -> Path:
    opp_median = metrics["opportunity_score"].median()
    pressure_median = metrics["rent_pressure"].median()

    def classify(row: pd.Series) -> str:
        high_opp = row["opportunity_score"] >= opp_median
        high_pressure = row["rent_pressure"] >= pressure_median
        if high_opp and high_pressure:
            return "高机会高压力"
        if high_opp and not high_pressure:
            return "高机会低压力"
        if not high_opp and high_pressure:
            return "低机会高压力"
        return "低机会低压力"

    plot_df = metrics.copy()
    plot_df["index_quadrant"] = plot_df.apply(classify, axis=1)
    palette = {
        "高机会高压力": "#d7191c",
        "高机会低压力": "#2c7bb6",
        "低机会高压力": "#fdae61",
        "低机会低压力": "#abd9e9",
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    for name, group in plot_df.groupby("index_quadrant"):
        ax.scatter(
            group["opportunity_score"],
            group["rent_pressure"],
            s=120 + group["ycsi"] * 5,
            color=palette[name],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
            label=name,
        )
        for _, row in group.iterrows():
            ax.text(row["opportunity_score"] + 0.012, row["rent_pressure"] + 0.012, row["city"], fontsize=8.5)
    ax.axvline(opp_median, color="#555555", linestyle="--", linewidth=1)
    ax.axhline(pressure_median, color="#555555", linestyle="--", linewidth=1)
    ax.text(0.77, 0.94, "高机会\n高压力", ha="center", va="center", fontsize=11, color="#8b0000")
    ax.text(0.77, 0.08, "高机会\n低压力", ha="center", va="center", fontsize=11, color="#0b4f6c")
    ax.set_xlabel("机会指数")
    ax.set_ylabel("居住压力指数（越高压力越大）")
    ax.set_title("城市机会-居住压力四象限")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="best")
    return save_figure("04_opportunity_pressure_quadrant.png")


def plot_city_radar(metrics: pd.DataFrame) -> Path:
    radar_map = {
        "机会": "opportunity_score",
        "居住友好": "rent_friendliness",
        "通勤友好": "commute_friendliness",
        "生活便利": "life_score",
        "成长潜力": "growth_score",
    }
    labels = list(radar_map.keys())
    columns = list(radar_map.values())

    selected: list[str] = []
    selected.extend(metrics.sort_values("ycsi", ascending=False)["city"].head(2).tolist())
    selected.extend(metrics.sort_values("rent_pressure", ascending=True)["city"].head(2).tolist())
    selected.extend(metrics.sort_values("growth_score", ascending=False)["city"].head(2).tolist())
    selected = list(dict.fromkeys(selected))[:5]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8.2, 8.2), subplot_kw={"polar": True})
    for city in selected:
        values = metrics.loc[metrics["city"] == city, columns].iloc[0].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=city)
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title("典型城市 YCSI 五维雷达图", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.08), frameon=False)
    return save_figure("05_city_radar.png")


def plot_poi_stack(metrics: pd.DataFrame, poi_scaled: pd.DataFrame) -> Path:
    display_names = {
        "subway": "地铁",
        "hospital": "医院",
        "park": "公园",
        "mall": "商场",
        "restaurant": "餐饮",
        "library": "图书馆",
        "gym": "健身房",
    }
    poi_components = poi_scaled.copy()
    poi_components.index = metrics["city"]
    poi_components = poi_components.loc[metrics.sort_values("life_score", ascending=False)["city"]]
    poi_components = poi_components.rename(columns=display_names)
    fig, ax = plt.subplots(figsize=(11, 7))
    poi_components.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c", width=0.72)
    ax.set_ylabel("POI 综合量（按总量标准化）")
    ax.set_xlabel("城市")
    ax.set_title("生活便利 POI 构成对比")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(title="POI 类型", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.xticks(rotation=35, ha="right")
    return save_figure("06_poi_convenience_stack.png")


def plot_correlation_heatmap(metrics: pd.DataFrame) -> Path:
    columns = [
        "ycsi",
        "avg_salary",
        "avg_rent",
        "rent_income_ratio",
        "job_count",
        "opportunity_score",
        "rent_pressure",
        "life_score",
        "growth_score",
        "commute_score",
    ]
    labels = [
        "YCSI",
        "月薪",
        "租金",
        "租金收入比",
        "岗位数",
        "机会",
        "居住压力",
        "生活便利",
        "成长",
        "通勤压力",
    ]
    corr = metrics[columns].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for row_idx in range(corr.shape[0]):
        for col_idx in range(corr.shape[1]):
            value = corr.iloc[row_idx, col_idx]
            color = "white" if abs(value) > 0.55 else "#222222"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_title("核心指标相关性热力图")
    color_bar = plt.colorbar(image, ax=ax, shrink=0.82)
    color_bar.set_label("Pearson 相关系数")
    return save_figure("07_metric_correlation_heatmap.png")


def plot_city_clusters(metrics: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    plot_df = metrics.copy()
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    plot_df["cluster_id"] = kmeans.fit_predict(plot_df[YCSI_FEATURES])
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=YCSI_FEATURES)

    cluster_names: dict[int, str] = {}
    used_names: dict[str, int] = {}
    center_median = centers.median()
    for cluster_id, row in centers.iterrows():
        opp_high = row["opportunity_score"] >= center_median["opportunity_score"]
        pressure_high = row["rent_pressure"] >= center_median["rent_pressure"]
        life_high = row["life_score"] >= center_median["life_score"]
        growth_high = row["growth_score"] >= center_median["growth_score"]
        commute_high = row["commute_score"] >= center_median["commute_score"]
        if opp_high and pressure_high and commute_high:
            name = "高机会高压力型"
        elif opp_high and not pressure_high:
            name = "机会友好型"
        elif opp_high and pressure_high:
            name = "机会成长型"
        elif (not opp_high) and (not pressure_high) and commute_high:
            name = "大城通勤压力型"
        elif growth_high and not pressure_high:
            name = "潜力成长型"
        elif life_high and not commute_high:
            name = "舒适均衡型"
        elif not pressure_high:
            name = "低压稳健型"
        else:
            name = "综合过渡型"
        used_names[name] = used_names.get(name, 0) + 1
        cluster_names[int(cluster_id)] = name if used_names[name] == 1 else f"{name}-{used_names[name]}"

    plot_df["cluster_name"] = plot_df["cluster_id"].map(cluster_names)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, plot_df["cluster_id"].nunique()))
    for color, (name, group) in zip(colors, plot_df.groupby("cluster_name")):
        ax.scatter(
            group["opportunity_score"],
            group["rent_pressure"],
            s=140 + group["ycsi"] * 4,
            color=color,
            alpha=0.84,
            edgecolor="white",
            linewidth=0.9,
            label=name,
        )
        for _, row in group.iterrows():
            ax.text(row["opportunity_score"] + 0.012, row["rent_pressure"] + 0.012, row["city"], fontsize=8.5)
    ax.set_xlabel("机会指数")
    ax.set_ylabel("居住压力指数（越高压力越大）")
    ax.set_title("K-Means 城市分类：机会与压力视角")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(alpha=0.22)
    ax.legend(title="聚类类型", frameon=False, loc="best")
    figure_path = save_figure("08_city_clusters.png")

    summary = (
        plot_df.groupby("cluster_name")
        .agg(
            city_list=("city", lambda values: "、".join(map(str, values))),
            avg_ycsi=("ycsi", "mean"),
            city_count=("city", "count"),
        )
        .round(2)
        .reset_index()
    )
    return figure_path, summary


def plot_rent_trend(metrics: pd.DataFrame) -> Path | None:
    """商品房租金 2018-2025 月度趋势（真实时间序列，补足时间维度）。"""
    panel_path = CLEAN_DIR / "rent_panel.csv"
    if not panel_path.exists():
        return None
    panel = read_csv_fallback(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["rent_per_sqm"] = pd.to_numeric(panel["rent_per_sqm"], errors="coerce")
    panel = panel.dropna(subset=["date", "rent_per_sqm"])

    # 高亮 YCSI 前 6 城，其余城市淡灰背景
    top_cities = metrics.sort_values("ycsi", ascending=False)["city"].head(6).tolist()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for city, grp in panel.groupby("city"):
        grp = grp.sort_values("date")
        if city in top_cities:
            ax.plot(grp["date"], grp["rent_per_sqm"], linewidth=2, label=city, zorder=3)
        else:
            ax.plot(grp["date"], grp["rent_per_sqm"], linewidth=0.8,
                    color="#cccccc", alpha=0.6, zorder=1)
    ax.set_xlabel("时间")
    ax.set_ylabel("商品房平均出租价格（元/㎡·月）")
    ax.set_title("15 城商品房租金月度趋势（2018-2025，高亮 YCSI 前 6）")
    ax.grid(alpha=0.22)
    ax.legend(title="YCSI 前 6 城", frameon=False, loc="upper left", ncol=2)
    return save_figure("09_rent_trend.png")


def export_scores(metrics: pd.DataFrame) -> Path:
    columns = [
        "city",
        "rank",
        "ycsi",
        "avg_salary",
        "avg_rent",
        "rent_income_ratio",
        "job_count",
        "opportunity_score",
        "rent_pressure",
        "life_score",
        "growth_score",
        "commute_score",
    ]
    score_df = metrics.sort_values("rank")[columns].reset_index(drop=True)
    path = OUTPUT_DIR / "ycsi_city_scores.csv"
    score_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_cluster_summary(summary: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "cluster_summary.csv"
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run() -> None:
    plan_path = write_chart_plan()
    metrics = load_base_data()
    metrics, source_notes = add_job_and_rent_metrics(metrics)
    metrics, poi_scaled = calculate_ycsi(metrics)

    generated = [
        plan_path,
        export_scores(metrics),
        plot_ycsi_ranking(metrics),
        plot_city_heat_points(metrics),
        plot_salary_rent_bubble(metrics),
        plot_opportunity_pressure_quadrant(metrics),
        plot_city_radar(metrics),
        plot_poi_stack(metrics, poi_scaled),
        plot_correlation_heatmap(metrics),
    ]
    cluster_path, cluster_summary = plot_city_clusters(metrics)
    generated.append(cluster_path)
    generated.append(export_cluster_summary(cluster_summary))

    trend_path = plot_rent_trend(metrics)
    if trend_path is not None:
        generated.append(trend_path)

    print("数据来源说明：")
    for note in source_notes:
        print(f"- {note}")
    print("\nYCSI Top 5：")
    print(metrics.sort_values("rank")[["rank", "city", "ycsi"]].head(5).to_string(index=False))
    print("\n已生成文件：")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    run()
