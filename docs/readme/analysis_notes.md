# Analysis Notes — Perth Crime Hotspot Analysis

## Data Source

The sample dataset (`data/raw/perth_crime_statistics.csv`) is modelled on the
**WA Police Force Crime Statistics** published on
[data.wa.gov.au](https://data.wa.gov.au/dataset/crime-statistics-by-suburb).

The real dataset can be downloaded as a CSV from the Open Data portal and
placed in `data/raw/` with the same column structure.

## Dataset Schema

| Column | Description |
|---|---|
| `Financial Year` | WA financial year, e.g. `2022/23` |
| `Period` | Quarter within the financial year (`Q1`–`Q4`) |
| `Suburb` | Perth suburb name |
| `District` | WA Police district |
| `Local Government Area` | LGA name |
| `Latitude` | Approximate latitude of the suburb centroid |
| `Longitude` | Approximate longitude of the suburb centroid |
| `Offence Division` | High-level offence category |
| `Offence Subdivision` | Specific offence type |
| `Offence Count` | Number of recorded offences for the period |

## Pipeline Steps

### Step 1 — Preprocessing (`01_preprocess.py`)
- Strips whitespace and standardises column names (lowercase + underscores).
- Enforces correct data types (numeric counts and coordinates).
- Drops rows with missing critical values.
- Builds a **suburb-level summary** by pivoting offence divisions into feature columns.

### Step 2 — Clustering (`02_clustering.py`)
- Scales features with `StandardScaler`.
- Evaluates K-Means for k = 2…8 using inertia (elbow) and silhouette scores.
- Applies K-Means with the chosen k (default 4).
- Assigns human-readable **risk tiers** ranked by mean total offences per cluster.
- Produces PCA scatter plot and elbow-curve PNG.

### Step 3 — Visualisation (`03_visualization.py`)
- **Folium HeatMap** — intensity weighted by total offences.
- **Folium Marker Map** — circles sized by offence count, coloured by risk tier.
- **Plotly Bar Chart** — top 15 suburbs by total offences.
- **Plotly Stacked Bar** — offence-type breakdown for top 10 suburbs.

## Limitations

- Coordinates are approximate suburb centroids; they do not reflect exact
  crime incident locations.
- The sample dataset covers two quarters of 2022/23. For a full analysis,
  download multiple years from data.wa.gov.au.
- K-Means assumes Euclidean distance and roughly spherical clusters.
  DBSCAN or hierarchical clustering may yield better results for geospatial
  point data.

## References

- WA Police Force Crime Statistics: https://www.police.wa.gov.au/Crime/CrimeStatistics
- Open Data WA: https://data.wa.gov.au
- Scikit-learn KMeans: https://scikit-learn.org/stable/modules/clustering.html#k-means
- Folium documentation: https://python-visualization.github.io/folium/
- Plotly documentation: https://plotly.com/python/
