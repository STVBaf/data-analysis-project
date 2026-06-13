# YCSI 可视化图表清单

| 图表 | 对应研究问题 | 主要字段 | 输出文件 |
|---|---|---|---|
| 全国城市生存力指数点状热力图 | 15 个目标城市的综合友好度与空间分布 | city, lng, lat, YCSI, population | 02_ycsi_city_heat_points.png |
| YCSI 排名柱状图 | 哪些城市更适合作为毕业后的第一座城市 | YCSI, opportunity, life, growth, rent pressure, commute pressure | 01_ycsi_ranking.png |
| 工资 vs 租金收入比气泡图 | 就业机会与租房压力是否匹配 | avg_salary, avg_rent, rent_income_ratio, job_count | 03_salary_rent_bubble.png |
| 机会-居住压力四象限图 | 城市属于高机会高压力、低压稳健等哪类 | opportunity_score, rent_pressure, YCSI | 04_opportunity_pressure_quadrant.png |
| 城市指标雷达图 | 典型城市在机会、居住、通勤、生活、成长上的差异 | five YCSI dimensions | 05_city_radar.png |
| 生活便利 POI 构成图 | 城市日常生活便利度差异 | subway, hospital, park, mall, restaurant, library, gym | 06_poi_convenience_stack.png |
| 指标相关性热力图 | 薪资、租金、生活便利、就业机会之间的关系 | salary, rent, job count, YCSI dimensions | 07_metric_correlation_heatmap.png |
| K-Means 城市聚类图 | 推荐不同类型青年适合的城市类别 | five YCSI dimensions | 08_city_clusters.png |

可选扩展：通勤-租房地图需要 `rent_data.csv` 中的区县/房源级租金、经纬度和就业中心数据；文本词云需要 `city_text.csv` 的 `city,text` 评论数据。
