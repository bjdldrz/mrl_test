"""
DynamicMission 数据集读取脚本
============================
用途：读取论文 MRL-DMS 配套发布的动态任务数据集（基于 ACLED 构建），
      该数据集以 ESRI Shapefile 格式存储（.shp/.shx/.dbf/.prj/.cpg 等文件组成一个整体，
      读取时只需指向 .shp 文件，其余同名文件会被自动关联读取）。

数据来源：
    GitHub : https://github.com/YYYauW/DynamicMission
    Zenodo : https://doi.org/10.5281/zenodo.17724850

依赖安装：
    pip install geopandas shapely pyproj fiona --break-system-packages

字段说明（参见 Data_Description.docx）：
    FID        - 内部行索引，无语义
    Frequency  - 该地点ACLED事件发生频率，用于建模动态任务到达强度
    Latitude / Longitude - WGS84坐标
    Landid     - 区域级数字标识符
    Region     - 大区域（如 East Asia, Europe ...）
    Country    - 国家
    Location   - 更细粒度地名
    Disorder   - ACLED高层级冲突类型（Political violence / Demonstrations）
    Event type - ACLED细分事件类型（Battles/Protests/Explosions/
                 Remote violence/Violence against civilians），
                 注意：shapefile的.dbf字段名最长10个字符，
                 "Event type" 在文件中可能被截断为 "Event_typ" 或 "EventType" 等，
                 本脚本会自动做模糊匹配兼容。
"""

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
# 1. 读取函数
# -----------------------------------------------------------------------
def load_dynamic_mission(shp_path: str) -> gpd.GeoDataFrame:
    """
    读取 DynamicMission.shp（及同目录下的 .shx/.dbf/.prj/.cpg 等配套文件）。

    参数
    ----
    shp_path : str
        指向 .shp 文件的路径，例如 "data/DynamicMission/DynamicMission.shp"

    返回
    ----
    geopandas.GeoDataFrame，列名已标准化为：
        FID, Frequency, Latitude, Longitude, Landid,
        Region, Country, Location, Disorder, EventType, geometry
    """
    if not os.path.exists(shp_path):
        raise FileNotFoundError(
            f"未找到文件: {shp_path}\n"
            "请确认 .shp/.shx/.dbf/.prj 等文件在同一目录下，且路径正确。"
        )

    gdf = gpd.read_file(shp_path)

    # --- 字段名标准化（兼容 shapefile 10字符截断导致的不同写法） ---
    rename_map = {}
    for col in gdf.columns:
        c_lower = col.lower().replace('_', '').replace(' ', '')
        if c_lower in ('fid',):
            rename_map[col] = 'FID'
        elif c_lower.startswith('freq'):
            rename_map[col] = 'Frequency'
        elif c_lower.startswith('lat'):
            rename_map[col] = 'Latitude'
        elif c_lower.startswith('lon') or c_lower.startswith('lng'):
            rename_map[col] = 'Longitude'
        elif c_lower.startswith('landid'):
            rename_map[col] = 'Landid'
        elif c_lower.startswith('region'):
            rename_map[col] = 'Region'
        elif c_lower.startswith('country'):
            rename_map[col] = 'Country'
        elif c_lower.startswith('location') or c_lower.startswith('loc'):
            rename_map[col] = 'Location'
        elif c_lower.startswith('disorder'):
            rename_map[col] = 'Disorder'
        elif c_lower.startswith('event'):
            rename_map[col] = 'EventType'

    gdf = gdf.rename(columns=rename_map)

    # 若坐标列缺失，但geometry是Point，则从geometry中补全经纬度
    if 'Latitude' not in gdf.columns or 'Longitude' not in gdf.columns:
        gdf['Longitude'] = gdf.geometry.x
        gdf['Latitude'] = gdf.geometry.y

    # 确保坐标系为 WGS84 (EPSG:4326)，与论文一致
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


# -----------------------------------------------------------------------
# 2. 基础数据探查
# -----------------------------------------------------------------------
def summarize(gdf: gpd.GeoDataFrame) -> None:
    print(f"记录总数: {len(gdf)}")
    print(f"字段列表: {list(gdf.columns)}")
    print("\n--- 前5行 ---")
    print(gdf.head())

    if 'Region' in gdf.columns:
        print("\n--- 按 Region 统计事件数 ---")
        print(gdf['Region'].value_counts())

    if 'EventType' in gdf.columns:
        print("\n--- 按 EventType 统计事件数 ---")
        print(gdf['EventType'].value_counts())

    if 'Frequency' in gdf.columns:
        print("\n--- Frequency 字段描述统计 ---")
        print(gdf['Frequency'].describe())


# -----------------------------------------------------------------------
# 3. 转换为调度实验所需的“动态任务”格式
#    （对应论文 Table 1: ID, Latitude, Longitude, Type, Profit, Location）
# -----------------------------------------------------------------------
def to_dynamic_missions(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    将原始事件数据转换为论文调度实验中使用的“动态任务”表，
    Frequency 作为任务回报（Profit）的代理变量。
    """
    cols = ['FID', 'Latitude', 'Longitude', 'Frequency',
            'Region', 'Country', 'Location', 'Disorder', 'EventType']
    cols = [c for c in cols if c in gdf.columns]
    df = gdf[cols].copy()
    df['Type'] = 'Dynamic'
    df = df.rename(columns={'Frequency': 'Profit'})
    return df


# -----------------------------------------------------------------------
# 4. 空间分布可视化（参考论文 Fig. 6 风格，按事件类型上色）
# -----------------------------------------------------------------------
def plot_spatial_distribution(gdf: gpd.GeoDataFrame, save_path: str = None) -> None:
    if 'EventType' not in gdf.columns:
        print("数据中无 EventType 字段，跳过分类绘图。")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for etype, sub in gdf.groupby('EventType'):
        ax.scatter(sub['Longitude'], sub['Latitude'],
                   s=sub['Frequency'] / sub['Frequency'].max() * 100 + 5
                   if 'Frequency' in sub.columns else 20,
                   alpha=0.6, label=etype)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Distribution of Dynamic Missions (by Event Type)')
    ax.legend(loc='lower left', fontsize=8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图已保存至: {save_path}")
    else:
        plt.show()


# -----------------------------------------------------------------------
# 5. 主程序示例
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # 修改为你本地实际的 .shp 文件路径
    SHP_PATH = "DynamicMission/DynamicMission.shp"

    gdf = load_dynamic_mission(SHP_PATH)
    summarize(gdf)

    # 导出为调度实验可直接使用的 CSV
    missions_df = to_dynamic_missions(gdf)
    missions_df.to_csv("dynamic_missions.csv", index=False, encoding='utf-8-sig')
    print("\n已导出: dynamic_missions.csv")

    # 绘制空间分布图（与论文 Fig. 6 类似）
    plot_spatial_distribution(gdf, save_path="dynamic_missions_spatial.png")