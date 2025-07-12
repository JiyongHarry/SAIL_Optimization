import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

# ----------------------------------------------------
# Load your transmission line data
# ----------------------------------------------------
line_data = pd.read_csv("EI Seed Project/TEXAS 123-BT params (line).csv")

# Sanity check: ensure From/To Bus Latitude and Longitude are correctly assigned
# Longitude should be in range [-107, -93], Latitude in [25, 37]


# ----------------------------------------------------
# Build a GeoDataFrame with line geometries (lon, lat)
# ----------------------------------------------------
def make_line(row):
    return LineString(
        [
            (row["From Bus Longitude"], row["From Bus Latitude"]),  # (x, y)
            (row["To Bus Longitude"], row["To Bus Latitude"]),
        ]
    )


lines_gdf = gpd.GeoDataFrame(
    line_data,
    geometry=line_data.apply(make_line, axis=1),
    crs="EPSG:4326",  # WGS84: longitude-latitude
)

# ----------------------------------------------------
# Load and filter US states shapefile to get Texas
# ----------------------------------------------------
state_shapefile_path = "EI Seed Project/cb_2018_us_state_20m/cb_2018_us_state_20m.shp"
states = gpd.read_file(state_shapefile_path).to_crs("EPSG:4326")
texas = states[states["STUSPS"] == "TX"]

# ----------------------------------------------------
# Plotting
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# Plot Texas border
texas.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2)

# Plot transmission lines
lines_gdf.plot(ax=ax, color="blue", linewidth=1)

# Annotate each line with line_num at its center
for idx, row in lines_gdf.iterrows():
    x, y = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(x, y, str(row["line_num"]), fontsize=6, color="red")

# Set map display limits to Texas
ax.set_xlim([-107, -93])
ax.set_ylim([25, 37])

# Final plot formatting
ax.set_title("Texas Transmission Lines Overlay")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")
plt.grid(True)
plt.tight_layout()
plt.show()
