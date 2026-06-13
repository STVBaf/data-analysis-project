"""
requirements.txt
----------------
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
jieba>=0.42.1
pydantic>=2.0.0

启动命令
--------
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import jieba.analyse
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# 配置区
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

EXPECTED_FILES = {
    "city_basic": "city_basic.csv",
    "job_data": "job_data.csv",
    "rent_data": "rent_data.csv",
    "poi_data": "poi_data.csv",
    "ycsi_base": "ycsi_base.csv",
}

REQUIRED_COLUMNS = {
    "city_basic": [
        "city",
        "gdp",
        "population",
        "disposable_income",
        "university_count",
        "metro_lines",
    ],
    "job_data": [
        "city",
        "keyword",
        "job_title",
        "salary_min",
        "salary_max",
        "experience",
        "education",
        "company_type",
    ],
    "rent_data": [
        "city",
        "district",
        "rent",
        "room_type",
        "area",
        "rent_type",
        "lng",
        "lat",
    ],
    "poi_data": [
        "city",
        "hospital",
        "park",
        "mall",
        "restaurant",
        "library",
        "gym",
        "subway",
    ],
    "ycsi_base": [
        "city",
        "opportunity_score",
        "rent_pressure",
        "life_score",
        "growth_score",
        "commute_score",
    ],
}

NUMERIC_COLUMNS = {
    "city_basic": [
        "gdp",
        "population",
        "disposable_income",
        "university_count",
        "metro_lines",
    ],
    "job_data": ["salary_min", "salary_max"],
    "rent_data": ["rent", "area", "lng", "lat"],
    "poi_data": [
        "hospital",
        "park",
        "mall",
        "restaurant",
        "library",
        "gym",
        "subway",
    ],
    "ycsi_base": [
        "opportunity_score",
        "rent_pressure",
        "life_score",
        "growth_score",
        "commute_score",
    ],
}

TEXT_COLUMNS = {
    name: [column for column in columns if column not in NUMERIC_COLUMNS.get(name, [])]
    for name, columns in REQUIRED_COLUMNS.items()
}

YCSI_FEATURES = [
    "opportunity_score",
    "rent_pressure",
    "life_score",
    "growth_score",
    "commute_score",
]

DEFAULT_WEIGHTS = {
    "opportunity": 0.3,
    "life": 0.2,
    "growth": 0.2,
    "rent": 0.2,
    "commute": 0.1,
}

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
    "Beijing": (116.4074, 39.9042),
    "Shanghai": (121.4737, 31.2304),
    "Guangzhou": (113.2644, 23.1291),
    "Shenzhen": (114.0579, 22.5431),
    "Hangzhou": (120.1551, 30.2741),
    "Nanjing": (118.7969, 32.0603),
    "Suzhou": (120.5853, 31.2989),
    "Chengdu": (104.0665, 30.5728),
    "Chongqing": (106.5516, 29.5630),
    "Wuhan": (114.3054, 30.5931),
    "Xian": (108.9398, 34.3416),
    "Xi'an": (108.9398, 34.3416),
    "Changsha": (112.9388, 28.2282),
    "Zhengzhou": (113.6254, 34.7466),
    "Tianjin": (117.2000, 39.1333),
    "Qingdao": (120.3826, 36.0671),
}


# =============================================================================
# Pydantic 响应模型区
# =============================================================================

class WeightConfig(BaseModel):
    opportunity: float = Field(..., description="机会维度权重")
    life: float = Field(..., description="生活便利维度权重")
    growth: float = Field(..., description="成长潜力维度权重")
    rent: float = Field(..., description="居住压力维度权重")
    commute: float = Field(..., description="通勤压力维度权重")


class DimensionScore(BaseModel):
    opportunity_score: float
    rent_pressure: float
    life_score: float
    growth_score: float
    commute_score: float


class CityMapItem(BaseModel):
    city: str
    lng: float
    lat: float
    gdp: float
    population: float
    disposable_income: float
    university_count: float
    metro_lines: float
    ycsi: float
    rank: int
    dimensions: DimensionScore


class MapSummary(BaseModel):
    city_count: int
    avg_ycsi: float
    top_city: Optional[str]
    weights: WeightConfig


class MapData(BaseModel):
    year: int
    summary: MapSummary
    cities: List[CityMapItem]


class MapResponse(BaseModel):
    code: int = 200
    data: MapData
    msg: str = "success"


class QuadrantItem(BaseModel):
    city: str
    avg_salary: float
    avg_rent: float
    rent_income_ratio: float
    job_count: int
    quadrant: str


class QuadrantData(BaseModel):
    industry: Optional[str]
    x_axis: str
    y_axis: str
    bubble: str
    cities: List[QuadrantItem]


class QuadrantResponse(BaseModel):
    code: int = 200
    data: QuadrantData
    msg: str = "success"


class ClusterGroup(BaseModel):
    cluster_id: int
    name: str
    city_count: int
    cities: List[str]
    dimension_averages: DimensionScore
    raw_dimension_averages: DimensionScore


class ClusterData(BaseModel):
    k_value: int
    feature_space: str
    clusters: List[ClusterGroup]


class ClusterResponse(BaseModel):
    code: int = 200
    data: ClusterData
    msg: str = "success"


class WordCloudItem(BaseModel):
    word: str
    weight: float
    sentiment: str


class WordCloudData(BaseModel):
    city: str
    top_k: int
    words: List[WordCloudItem]


class WordCloudResponse(BaseModel):
    code: int = 200
    data: WordCloudData
    msg: str = "success"


class ErrorResponse(BaseModel):
    code: int
    data: Dict[str, Any]
    msg: str


# =============================================================================
# 数据加载区
# =============================================================================

@dataclass
class DataStore:
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class ServiceError(Exception):
    def __init__(self, msg: str, code: int = 500, data: Optional[Dict[str, Any]] = None):
        self.code = code
        self.msg = msg
        self.data = data or {}
        super().__init__(msg)


def standard_response(data: Any, msg: str = "success", code: int = 200) -> Dict[str, Any]:
    return {"code": code, "data": data, "msg": msg}


def error_response(msg: str, code: int = 500, data: Optional[Dict[str, Any]] = None) -> JSONResponse:
    status_code = code if 400 <= code <= 599 else 200
    return JSONResponse(status_code=status_code, content=standard_response(data or {}, msg, code))


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def clean_dataframe(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]

    missing = [column for column in REQUIRED_COLUMNS[name] if column not in df.columns]
    if missing:
        raise ValueError(f"{EXPECTED_FILES[name]} 缺少字段: {', '.join(missing)}")

    keep_columns = REQUIRED_COLUMNS[name] + [
        column for column in df.columns if column not in REQUIRED_COLUMNS[name]
    ]
    df = df[keep_columns]

    for column in TEXT_COLUMNS.get(name, []):
        df[column] = df[column].fillna("").astype(str).str.strip()

    for column in NUMERIC_COLUMNS.get(name, []):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        fill_value = df[column].median(skipna=True)
        if pd.isna(fill_value):
            fill_value = 0.0
        df[column] = df[column].replace([np.inf, -np.inf], np.nan).fillna(float(fill_value))

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df


def load_data_store(data_dir: Path) -> DataStore:
    store = DataStore()

    for name, filename in EXPECTED_FILES.items():
        path = data_dir / filename
        try:
            if not path.exists():
                raise FileNotFoundError(f"缺少数据文件: {path}")
            raw_df = read_csv_with_fallback(path)
            store.frames[name] = clean_dataframe(name, raw_df)
        except Exception as exc:
            store.errors.append(str(exc))

    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.data_store = load_data_store(DATA_DIR)
    yield


# =============================================================================
# FastAPI 应用与全局异常处理区
# =============================================================================

app = FastAPI(
    title="青年城市生存力指数 YCSI 后端服务",
    description="毕业后的第一座城市：青年城市生存力指数可视化分析 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError) -> JSONResponse:
    return error_response(exc.msg, exc.code, exc.data)


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(_: Request, exc: FileNotFoundError) -> JSONResponse:
    return error_response("数据文件缺失", 500, {"detail": str(exc)})


@app.exception_handler(ZeroDivisionError)
async def zero_division_handler(_: Request, exc: ZeroDivisionError) -> JSONResponse:
    return error_response("计算过程中出现除零错误", 500, {"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response("请求参数校验失败", 422, {"errors": exc.errors()})


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return error_response("服务器内部错误", 500, {"detail": str(exc)})


# =============================================================================
# 服务逻辑区
# =============================================================================

def get_store() -> DataStore:
    store = getattr(app.state, "data_store", None)
    if store is None:
        raise ServiceError("数据缓存尚未初始化", 500)
    return store


def get_frame(name: str) -> pd.DataFrame:
    store = get_store()
    if name not in store.frames:
        raise ServiceError(
            f"数据未加载: {EXPECTED_FILES.get(name, name)}",
            500,
            {"load_errors": store.errors},
        )
    return store.frames[name].copy()


def filter_by_year(df: pd.DataFrame, year: Optional[int]) -> pd.DataFrame:
    if year is None or "year" not in df.columns:
        return df
    filtered = df[df["year"] == year].copy()
    return filtered if not filtered.empty else df


def round_float(value: Any, ndigits: int = 4) -> float:
    if value is None or pd.isna(value):
        return 0.0
    value = float(value)
    if np.isinf(value):
        return 0.0
    return round(value, ndigits)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def scaled_feature_frame(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    if df.empty:
        raise ServiceError("没有可用于标准化的数据", 404)

    matrix = df[features].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(matrix)
    return pd.DataFrame(scaled, columns=features, index=df.index)


def resolve_weights(
    w_opp: Optional[float],
    w_life: Optional[float],
    w_growth: Optional[float],
    w_rent: Optional[float],
    w_com: Optional[float],
) -> Dict[str, float]:
    return {
        "opportunity": DEFAULT_WEIGHTS["opportunity"] if w_opp is None else float(w_opp),
        "life": DEFAULT_WEIGHTS["life"] if w_life is None else float(w_life),
        "growth": DEFAULT_WEIGHTS["growth"] if w_growth is None else float(w_growth),
        "rent": DEFAULT_WEIGHTS["rent"] if w_rent is None else float(w_rent),
        "commute": DEFAULT_WEIGHTS["commute"] if w_com is None else float(w_com),
    }


def calculate_ycsi(scaled: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    return (
        scaled["opportunity_score"] * weights["opportunity"]
        + scaled["life_score"] * weights["life"]
        + scaled["growth_score"] * weights["growth"]
        - scaled["rent_pressure"] * weights["rent"]
        - scaled["commute_score"] * weights["commute"]
    )


def get_city_coord(city: str) -> Dict[str, float]:
    if city in CITY_COORDS:
        lng, lat = CITY_COORDS[city]
        return {"lng": lng, "lat": lat}

    store = get_store()
    rent_df = store.frames.get("rent_data")
    if rent_df is not None and not rent_df.empty:
        rows = rent_df[rent_df["city"] == city]
        rows = rows[(rows["lng"] != 0) & (rows["lat"] != 0)]
        if not rows.empty:
            return {
                "lng": round_float(rows["lng"].mean(), 6),
                "lat": round_float(rows["lat"].mean(), 6),
            }

    return {"lng": 0.0, "lat": 0.0}


def classify_quadrant(avg_salary: float, ratio: float, salary_median: float, ratio_median: float) -> str:
    high_salary = avg_salary >= salary_median
    high_pressure = ratio >= ratio_median

    if high_salary and high_pressure:
        return "高薪高压"
    if high_salary and not high_pressure:
        return "高薪宜居"
    if not high_salary and high_pressure:
        return "低薪高压"
    return "低压稳健"


def build_cluster_names(center_df: pd.DataFrame) -> Dict[int, str]:
    medians = center_df.median()
    names: Dict[int, str] = {}
    used: Dict[str, int] = {}

    for cluster_id, row in center_df.iterrows():
        opp_high = row["opportunity_score"] >= medians["opportunity_score"]
        rent_high = row["rent_pressure"] >= medians["rent_pressure"]
        life_high = row["life_score"] >= medians["life_score"]
        growth_high = row["growth_score"] >= medians["growth_score"]
        commute_high = row["commute_score"] >= medians["commute_score"]

        if opp_high and rent_high:
            base_name = "高薪高压型"
        elif growth_high and not rent_high:
            base_name = "潜力成长型"
        elif life_high and not rent_high and not commute_high:
            base_name = "舒适均衡型"
        elif not rent_high and not commute_high:
            base_name = "低成本友好型"
        elif rent_high and not opp_high:
            base_name = "高压谨慎型"
        else:
            base_name = "综合均衡型"

        used[base_name] = used.get(base_name, 0) + 1
        names[int(cluster_id)] = base_name if used[base_name] == 1 else f"{base_name}-{used[base_name]}"

    return names


def sentiment_of_word(word: str) -> str:
    positive_words = {
        "包容",
        "机会",
        "活力",
        "便利",
        "地铁",
        "美食",
        "公园",
        "创新",
        "成长",
        "安全",
        "友好",
        "多元",
        "舒服",
        "宜居",
        "热情",
        "开放",
        "发展",
        "年轻",
    }
    negative_words = {
        "拥挤",
        "通勤",
        "压力",
        "房租",
        "内卷",
        "焦虑",
        "堵车",
        "昂贵",
        "嘈杂",
        "疲惫",
        "竞争",
        "加班",
        "孤独",
        "潮湿",
        "成本",
    }

    if word in positive_words:
        return "positive"
    if word in negative_words:
        return "negative"
    return "neutral"


def simulated_city_text(city: str) -> str:
    city_profiles = {
        "北京": "机会 创新 互联网 高校 博物馆 通勤 拥挤 房租 压力 成长",
        "上海": "开放 多元 金融 时尚 地铁 精致 昂贵 房租 竞争 机会",
        "广州": "包容 美食 烟火气 外贸 通勤 便利 成本 机会 热情",
        "深圳": "年轻 创新 科技 加班 竞争 成长 机会 房租 压力",
        "杭州": "互联网 公园 西湖 创新 宜居 成长 房租 通勤",
        "南京": "高校 稳定 历史 生活 便利 地铁 宜居 成长",
        "苏州": "产业 园区 舒服 宜居 通勤 成本 机会 制造业",
        "成都": "美食 公园 休闲 包容 生活 宜居 机会 成本",
        "重庆": "山城 美食 热情 活力 通勤 轻轨 潮湿 成本",
        "武汉": "高校 活力 交通 成长 美食 通勤 机会",
        "西安": "历史 高校 成本 稳定 成长 机会 生活",
        "长沙": "娱乐 美食 年轻 活力 成本 机会 热情",
        "郑州": "交通 成本 成长 稳定 机会 通勤",
        "天津": "稳定 生活 成本 地铁 宜居 机会",
        "青岛": "海边 宜居 生活 舒服 旅游 成本 机会",
    }
    base = (
        f"{city} 对毕业生来说有机会也有压力。很多人关注岗位数量、平均薪资、房租、通勤、"
        f"地铁便利、生活成本、城市包容度、成长空间和社交氛围。"
        f"有人觉得{city}年轻有活力，也有人担心房租昂贵、竞争激烈、加班和孤独。"
    )
    return " ".join([base, city_profiles.get(city, "机会 成长 生活 成本 通勤 便利 压力 包容")])


# =============================================================================
# 路由区
# =============================================================================

@app.get("/", tags=["Health"])
async def root() -> Dict[str, Any]:
    store = get_store()
    return standard_response(
        {
            "service": "YCSI API",
            "docs": "/docs",
            "loaded_tables": sorted(store.frames.keys()),
            "load_errors": store.errors,
        }
    )


@app.get("/api/ycsi/map", response_model=MapResponse, tags=["YCSI"])
async def get_ycsi_map(
    year: int = Query(2023, description="年份；当前 CSV 无 year 字段时仅原样返回"),
    w_opp: Optional[float] = Query(None, ge=0, description="机会维度权重"),
    w_life: Optional[float] = Query(None, ge=0, description="生活便利维度权重"),
    w_growth: Optional[float] = Query(None, ge=0, description="成长潜力维度权重"),
    w_rent: Optional[float] = Query(None, ge=0, description="居住压力维度权重"),
    w_com: Optional[float] = Query(None, ge=0, description="通勤压力维度权重"),
) -> Dict[str, Any]:
    city_basic = filter_by_year(get_frame("city_basic"), year)
    ycsi_base = filter_by_year(get_frame("ycsi_base"), year)
    weights = resolve_weights(w_opp, w_life, w_growth, w_rent, w_com)

    merged = city_basic.merge(ycsi_base, on="city", how="inner")
    if merged.empty:
        raise ServiceError("city_basic 与 ycsi_base 没有可匹配的城市数据", 404)

    scaled = scaled_feature_frame(merged, YCSI_FEATURES)
    for feature in YCSI_FEATURES:
        merged[f"{feature}_scaled"] = scaled[feature].values

    merged["ycsi"] = calculate_ycsi(scaled, weights)
    merged = merged.sort_values("ycsi", ascending=False).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)

    cities: List[Dict[str, Any]] = []
    for idx, row in merged.iterrows():
        coord = get_city_coord(str(row["city"]))
        city_dimensions = {
            feature: round_float(row[f"{feature}_scaled"])
            for feature in YCSI_FEATURES
        }
        cities.append(
            {
                "city": str(row["city"]),
                "lng": round_float(coord["lng"], 6),
                "lat": round_float(coord["lat"], 6),
                "gdp": round_float(row["gdp"], 2),
                "population": round_float(row["population"], 2),
                "disposable_income": round_float(row["disposable_income"], 2),
                "university_count": round_float(row["university_count"], 2),
                "metro_lines": round_float(row["metro_lines"], 2),
                "ycsi": round_float(row["ycsi"]),
                "rank": int(row["rank"]),
                "dimensions": city_dimensions,
            }
        )

    summary = {
        "city_count": len(cities),
        "avg_ycsi": round_float(merged["ycsi"].mean()),
        "top_city": cities[0]["city"] if cities else None,
        "weights": weights,
    }
    return standard_response({"year": year, "summary": summary, "cities": cities})


@app.get("/api/ycsi/quadrant", response_model=QuadrantResponse, tags=["YCSI"])
async def get_ycsi_quadrant(
    industry: Optional[str] = Query(None, description="按 job_data.keyword 过滤行业/岗位关键词")
) -> Dict[str, Any]:
    job_df = get_frame("job_data")
    rent_df = get_frame("rent_data")

    if industry:
        keyword = industry.strip()
        job_df = job_df[
            job_df["keyword"].astype(str).str.contains(keyword, case=False, na=False, regex=False)
        ]

    if job_df.empty:
        return standard_response(
            {
                "industry": industry,
                "x_axis": "avg_salary",
                "y_axis": "rent_income_ratio",
                "bubble": "job_count",
                "cities": [],
            }
        )

    job_df["avg_salary"] = (job_df["salary_min"] + job_df["salary_max"]) / 2.0
    salary_city = (
        job_df.groupby("city", as_index=False)
        .agg(avg_salary=("avg_salary", "mean"), job_count=("job_title", "count"))
    )
    rent_city = rent_df.groupby("city", as_index=False).agg(avg_rent=("rent", "mean"))
    merged = salary_city.merge(rent_city, on="city", how="inner")

    if merged.empty:
        raise ServiceError("招聘数据与租房数据没有可匹配的城市", 404)

    merged["rent_income_ratio"] = merged.apply(
        lambda row: safe_divide(row["avg_rent"], row["avg_salary"]),
        axis=1,
    )
    salary_median = float(merged["avg_salary"].median())
    ratio_median = float(merged["rent_income_ratio"].median())
    merged = merged.sort_values(["avg_salary", "rent_income_ratio"], ascending=[False, True])

    cities = []
    for _, row in merged.iterrows():
        cities.append(
            {
                "city": str(row["city"]),
                "avg_salary": round_float(row["avg_salary"], 2),
                "avg_rent": round_float(row["avg_rent"], 2),
                "rent_income_ratio": round_float(row["rent_income_ratio"], 4),
                "job_count": int(row["job_count"]),
                "quadrant": classify_quadrant(
                    float(row["avg_salary"]),
                    float(row["rent_income_ratio"]),
                    salary_median,
                    ratio_median,
                ),
            }
        )

    return standard_response(
        {
            "industry": industry,
            "x_axis": "avg_salary",
            "y_axis": "rent_income_ratio",
            "bubble": "job_count",
            "cities": cities,
        }
    )


@app.get("/api/analysis/cluster", response_model=ClusterResponse, tags=["Analysis"])
async def get_cluster_analysis(
    k_value: int = Query(4, ge=2, le=10, description="K-Means 聚类数量")
) -> Dict[str, Any]:
    ycsi_base = get_frame("ycsi_base")
    ycsi_base = ycsi_base.drop_duplicates(subset=["city"]).reset_index(drop=True)

    if len(ycsi_base) < k_value:
        raise ServiceError(
            "k_value 不能大于可聚类城市数量",
            400,
            {"city_count": len(ycsi_base), "k_value": k_value},
        )

    scaled = scaled_feature_frame(ycsi_base, YCSI_FEATURES)
    model = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    labels = model.fit_predict(scaled[YCSI_FEATURES])

    ycsi_base["cluster_id"] = labels
    scaled_with_cluster = scaled.copy()
    scaled_with_cluster["cluster_id"] = labels

    center_df = pd.DataFrame(model.cluster_centers_, columns=YCSI_FEATURES)
    cluster_names = build_cluster_names(center_df)

    clusters: List[Dict[str, Any]] = []
    for cluster_id in sorted(ycsi_base["cluster_id"].unique()):
        raw_rows = ycsi_base[ycsi_base["cluster_id"] == cluster_id]
        scaled_rows = scaled_with_cluster[scaled_with_cluster["cluster_id"] == cluster_id]
        dimension_averages = {
            feature: round_float(scaled_rows[feature].mean())
            for feature in YCSI_FEATURES
        }
        raw_dimension_averages = {
            feature: round_float(raw_rows[feature].mean())
            for feature in YCSI_FEATURES
        }
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "name": cluster_names[int(cluster_id)],
                "city_count": int(len(raw_rows)),
                "cities": raw_rows["city"].astype(str).tolist(),
                "dimension_averages": dimension_averages,
                "raw_dimension_averages": raw_dimension_averages,
            }
        )

    return standard_response(
        {
            "k_value": k_value,
            "feature_space": "MinMaxScaler normalized 5 YCSI dimensions",
            "clusters": clusters,
        }
    )


@app.get("/api/city/wordcloud", response_model=WordCloudResponse, tags=["City"])
async def get_city_wordcloud(
    city: str = Query(..., min_length=1, description="城市名称")
) -> Dict[str, Any]:
    city = city.strip()
    text = simulated_city_text(city)
    tags = jieba.analyse.extract_tags(text, topK=30, withWeight=True)

    if not tags:
        return standard_response({"city": city, "top_k": 30, "words": []})

    max_weight = max(weight for _, weight in tags) or 1.0
    words = [
        {
            "word": word,
            "weight": round_float(weight / max_weight, 4),
            "sentiment": sentiment_of_word(word),
        }
        for word, weight in tags
    ]

    return standard_response({"city": city, "top_k": 30, "words": words})
