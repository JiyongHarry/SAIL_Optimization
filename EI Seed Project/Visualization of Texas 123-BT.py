import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from shapely.geometry import LineString
from collections import defaultdict
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import matplotlib.widgets as mwidgets

# ----------------------------------------------------
# Load transmission line data
# ----------------------------------------------------
line_data = pd.read_csv("EI Seed Project/TEXAS 123-BT params (line).csv")


# ----------------------------------------------------
# Create GeoDataFrame with line geometries
# ----------------------------------------------------
def make_line(row):
    return LineString(
        [
            (row["From Bus Longitude"], row["From Bus Latitude"]),
            (row["To Bus Longitude"], row["To Bus Latitude"]),
        ]
    )


lines_gdf = gpd.GeoDataFrame(
    line_data, geometry=line_data.apply(make_line, axis=1), crs="EPSG:4326"
)

# ----------------------------------------------------
# Load Texas boundary
# ----------------------------------------------------
state_shapefile_path = "EI Seed Project/cb_2018_us_state_20m/cb_2018_us_state_20m.shp"
states = gpd.read_file(state_shapefile_path).to_crs("EPSG:4326")
texas = states[states["STUSPS"] == "TX"]

# Extract unique bus locations from both From and To buses
bus_locs = pd.concat(
    [
        line_data[
            ["From Bus Number", "From Bus Longitude", "From Bus Latitude"]
        ].rename(
            columns={
                "From Bus Number": "Bus Number",
                "From Bus Longitude": "Longitude",
                "From Bus Latitude": "Latitude",
            }
        ),
        line_data[["To Bus Number", "To Bus Longitude", "To Bus Latitude"]].rename(
            columns={
                "To Bus Number": "Bus Number",
                "To Bus Longitude": "Longitude",
                "To Bus Latitude": "Latitude",
            }
        ),
    ]
).drop_duplicates(subset=["Bus Number"])

# ----------------------------------------------------
# Plotting
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))
texas.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2)

# Plot lines and keep reference
lines_gdf.plot(ax=ax, color="blue", linewidth=1, label="Line")

# Get the LineCollection object for blue transmission lines only
trans_line_collection = None
for c in ax.collections:
    if isinstance(c, LineCollection):
        # Check if the color is blue (in RGBA)
        facecolors = c.get_facecolor()
        edgecolors = c.get_edgecolor()
        # Both facecolors and edgecolors are arrays of RGBA
        # Blue in RGBA is (0, 0, 1, 1)
        if (facecolors.size > 0 and (facecolors[0][:3] == (0, 0, 1)).all()) or (
            edgecolors.size > 0 and (edgecolors[0][:3] == (0, 0, 1)).all()
        ):
            trans_line_collection = c
            break
if trans_line_collection is None:
    # Fallback: just use the first LineCollection (should not happen if blue lines exist)
    trans_line_collection = [
        c for c in ax.collections if isinstance(c, LineCollection)
    ][0]

# Plot buses and keep reference
bus_plot = ax.scatter(
    bus_locs["Longitude"],
    bus_locs["Latitude"],
    color="green",
    s=30,
    zorder=5,
    label="Bus",
    clip_on=True,  # Ensure dots are clipped to axes
)

# Annotate each bus with its bus number in green, keep references to text objects
bus_texts = []
for _, row in bus_locs.iterrows():
    t = ax.text(
        row["Longitude"] + 0.05,  # small offset to the right
        row["Latitude"] + 0.05,  # small offset up
        str(int(row["Bus Number"])),
        fontsize=8,
        color="green",
        weight="bold",
        zorder=6,
        bbox=dict(
            facecolor="white", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.1"
        ),
        clip_on=True,  # Ensure text is clipped to axes
    )
    bus_texts.append(t)

# Annotate each line with its line number, offset perpendicularly to the line to avoid overlap, keep references to text objects
line_texts = []
for idx, row in lines_gdf.iterrows():
    line = row.geometry
    centroid = line.centroid
    x, y = centroid.x, centroid.y

    # Calculate a perpendicular offset direction
    x0, y0 = line.coords[0]
    x1, y1 = line.coords[-1]
    dx = x1 - x0
    dy = y1 - y0
    length = (dx**2 + dy**2) ** 0.5
    if length == 0:
        offset_x, offset_y = 0, 0.05
    else:
        # Perpendicular direction (normalized)
        offset_x = -dy / length * 0.08  # adjust 0.08 as needed
        offset_y = dx / length * 0.08

    t = ax.text(
        x + offset_x,
        y + offset_y,
        str(row["line_num"]),
        fontsize=7,  # Increased font size
        color="red",
        weight="bold",
        bbox=dict(
            facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.1"
        ),
        clip_on=True,  # Ensure text is clipped to axes
    )
    line_texts.append(t)

# --- Add major Texas cities (updated: only Dallas in Dallas area, add West Texas cities) ---
major_cities = [
    {"name": "Houston", "lon": -95.3698, "lat": 29.7604},
    {"name": "San Antonio", "lon": -98.4936, "lat": 29.4241},
    {"name": "Dallas", "lon": -96.7970, "lat": 32.7767},
    {"name": "Austin", "lon": -97.7431, "lat": 30.2672},
    {"name": "El Paso", "lon": -106.4850, "lat": 31.7619},
    {"name": "Corpus Christi", "lon": -97.3964, "lat": 27.8006},
    {"name": "Laredo", "lon": -99.4803, "lat": 27.5306},
    {"name": "Lubbock", "lon": -101.8552, "lat": 33.5779},
    {"name": "Midland", "lon": -102.0779, "lat": 32.0005},
    {"name": "Odessa", "lon": -102.3676, "lat": 31.8457},
    {"name": "Amarillo", "lon": -101.8313, "lat": 35.2219},
]

city_lons = [c["lon"] for c in major_cities]
city_lats = [c["lat"] for c in major_cities]
city_names = [c["name"] for c in major_cities]


# --- Make major cities toggleable in the interactive legend panel ---
city_star_plot = ax.scatter(
    city_lons,
    city_lats,
    color="orange",
    marker="*",
    s=120,
    zorder=10,
    label="Major City",
)

# Annotate city names and keep references
city_texts = []
for lon, lat, name in zip(city_lons, city_lats, city_names):
    t = ax.text(
        lon + 0.08,
        lat + 0.08,
        name,
        fontsize=9,
        color="orange",
        weight="bold",
        zorder=11,
        bbox=dict(
            facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.1"
        ),
        clip_on=True,
    )
    city_texts.append(t)


# Add interactive legend panel below the plot
rax = plt.axes(
    [0.4, 0.02, 0.25, 0.12]
)  # [left, bottom, width, height] (bottom center, slightly larger)
labels = ["Lines", "Buses", "Major Cities"]
visibility = [
    trans_line_collection.get_visible(),
    bus_plot.get_visible(),
    city_star_plot.get_visible(),
]
check = CheckButtons(rax, labels, visibility)


# ----------------------------------------------------
# Plot transmission lines as arrows to show direction (with relative head size)
# ----------------------------------------------------
arrow_artists = []
x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
head_width_rel = 0.01 * x_range  # 1% of x axis range
head_length_rel = 0.015 * y_range  # 1.5% of y axis range

for _, row in line_data.iterrows():
    x_start, y_start = row["From Bus Longitude"], row["From Bus Latitude"]
    x_end, y_end = row["To Bus Longitude"], row["To Bus Latitude"]
    arrow = FancyArrowPatch(
        (x_start, y_start),
        (x_end, y_end),
        arrowstyle="-|>",
        color="blue",
        alpha=0.7,
        linewidth=1,
        zorder=3,
        mutation_scale=20,  # Try 20-40 for visibility; increase if still too small
        transform=ax.transData,
    )
    ax.add_patch(arrow)
    arrow_artists.append(arrow)


# Update the interactive legend callback to include arrows
def func(label):
    if label == "Lines":
        vis = not trans_line_collection.get_visible()
        trans_line_collection.set_visible(vis)
        for t in line_texts:
            t.set_visible(vis)
        for arr in arrow_artists:
            arr.set_visible(vis)
    elif label == "Buses":
        vis = not bus_plot.get_visible()
        bus_plot.set_visible(vis)
        for t in bus_texts:
            t.set_visible(vis)
    elif label == "Major Cities":
        vis = not city_star_plot.get_visible()
        city_star_plot.set_visible(vis)
        for t in city_texts:
            t.set_visible(vis)
    plt.draw()


check.on_clicked(func)

# --- Add a "Find Bus" text box to jump to a bus location ---
from matplotlib.widgets import TextBox


def find_and_center_bus(text):
    try:
        bus_num = int(text)
        bus_row = bus_locs[bus_locs["Bus Number"] == bus_num]
        if not bus_row.empty:
            lon = bus_row["Longitude"].values[0]
            lat = bus_row["Latitude"].values[0]
            # Set new limits centered on the bus, with a small window
            ax.set_xlim(lon - 0.5, lon + 0.5)
            ax.set_ylim(lat - 0.5, lat + 0.5)
            plt.draw()
        else:
            print(f"Bus {bus_num} not found.")
    except Exception as e:
        print(f"Invalid input: {e}")


# Place the TextBox below the plot (adjust position as needed)
find_box_ax = plt.axes([0.7, 0.02, 0.15, 0.05])  # [left, bottom, width, height]
find_box = TextBox(find_box_ax, "Find Bus #", initial="")
find_box.on_submit(find_and_center_bus)

# Store the original Texas view limits
original_xlim = [-107, -93]
original_ylim = [25, 37]
ax.set_xlim(original_xlim)
ax.set_ylim(original_ylim)


# Connect the home button to reset the view to the original Texas level
def on_home(event):
    ax.set_xlim(original_xlim)
    ax.set_ylim(original_ylim)
    plt.draw()


fig = plt.gcf()
toolbar = plt.get_current_fig_manager().toolbar
if hasattr(toolbar, "home"):
    toolbar.home = on_home  # For some backends
else:
    # For most matplotlib backends, connect to the 'home' event
    fig.canvas.mpl_connect("home_event", on_home)

# Set plot bounds for Texas
ax.set_xlim([-107, -93])
ax.set_ylim([25, 37])
ax.set_title("Texas 123-BT")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")
plt.grid(True)
plt.tight_layout()

plt.show()
