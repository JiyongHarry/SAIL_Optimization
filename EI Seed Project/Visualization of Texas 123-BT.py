import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

# Load your line data CSV
line_data = pd.read_csv("EI Seed Project/TEXAS 123-BT params (line).csv")

# Build a GeoDataFrame of line geometries
def make_line(row):
    return LineString([
        (row["From Bus Longitude"], row["From Bus Latitude"]),
        (row["To Bus Longitude"], row["To Bus Latitude"])
    ])

lines_gdf = gpd.GeoDataFrame(
    line_data,
    geometry=line_data.apply(make_line, axis=1),
    crs="EPSG:4326"
)

# Load US states shapefile from Census Bureau
state_shapefile_path = "cb_2018_us_state_20m.shp"  # update path
states = gpd.read_file(state_shapefile_path).to_crs("EPSG:4326")

# Filter Texas by two-letter postal code STUSPS == 'TX'
texas = states[states["STUSPS"] == "TX"]

# Plot map
fig, ax = plt.subplots(figsize=(10, 10))
texas.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2)
lines_gdf.plot(ax=ax, color="blue", linewidth=1)

# Annotate each line with line number
for idx, row in lines_gdf.iterrows():
    x, y = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(x, y, str(row["line_num"]), fontsize=6, color="red")

ax.set_title("Texas Transmission Lines Overlay")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
