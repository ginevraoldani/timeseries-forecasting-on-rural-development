import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
ERRORS_DIR = os.path.join(RESULTS_DIR, "errors")

RAW_DATA_FILE = os.path.join(DATA_DIR, "processed_italy_data.xlsx")
PERFORMANCE_FILE = os.path.join(ERRORS_DIR, "model_performances.xlsx")
RESIDUALS_FILE = os.path.join(ERRORS_DIR, "residuals_stats.xlsx")
PARAMS_FILE = os.path.join(LOGS_DIR, "optimized_params.xlsx")
PREDICTIONS_FILE = os.path.join(LOGS_DIR, "predictions_sequences.csv")

RANDOM_SEED = 42

UNUSABLE = [
    "Rural population living in areas where elevation is below 5 meters (% of total population)",
    "Access to electricity, rural (% of rural population)",
    "Surface area (sq. km)",
    "Rural land area (sq. km)",
    "Land area (sq. km)",
    "Average precipitation in depth (mm per year)",
    "Agricultural irrigated land (% of total agricultural land)",
    "Rural land area where elevation is below 5 meters (% of total land area)",
    "Rural land area where elevation is below 5 meters (sq. km)"
]

SAFE_VAR_NAME = {
    "Rural population (% of total population)" : "population_percent",
    "Rural population growth (annual %)" : "population_growth",
    "Rural population" : "population_abs",
    "Employment in agriculture (% of total employment)" : "employment_tot",
    "Employment in agriculture, male (% of male employment)" : "employment_male",
    "Employment in agriculture, female (% of female employment)" : "employment_female",
    "Forest area (% of land area)" : "forestarea_percent",
    "Forest area (sq. km)" : "forestarea_abs",
    "Agricultural land (% of land area)" : "agriland_percent",
    "Agricultural land (sq. km)" : "agriland_abs",
    "Arable land (% of land area)" : "arableland_percent",
    "Arable land (hectares per person)" : "arableland_person",
    "Arable land (hectares)" : "arableland_abs",
    "Land under cereal production (hectares)" : "cerealland_abs",
    "Permanent cropland (% of land area)" : "cropland_percent",
    "Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)" : "withdrawals_percent",
    "Fertilizer consumption (kilograms per hectare of arable land)" : "fertilizer_abs",
    "Fertilizer consumption (% of fertilizer production)" : "fertilizer_percent",
    "Livestock production index (2014-2016 = 100)" : "livestock_production_index",
    "Food production index (2014-2016 = 100)" : "food_production_index",
    "Crop production index (2014-2016 = 100)" : "crop_production_index",
    "Cereal production (metric tons)" : "cereal_production",
    "Cereal yield (kg per hectare)" : "cerealyield_abs",
    "Agriculture, forestry, and fishing, value added (% of GDP)" : "valueadded_percent",
    "Agriculture, forestry, and fishing, value added (current US$)" : "valueadded_dollars",
    "Agricultural raw materials exports (% of merchandise exports)" : "exports_percent",
    "Agricultural raw materials imports (% of merchandise imports)" : "imports_percent"    
}

COLORS = {
    'train':        '#4a5568',
    'test_real':    '#000000',
    'pred_baseline1': '#94a3b8',
    'pred_baseline2': '#64748b',
    'pred_model':     '#dc2626',
    'pred_orig':      '#2563eb',
    'pred_step':      '#16a34a',
    'pred_jitter':    '#ea580c'
}


LINE_STYLES = {
    'real':     '-',
    'baseline': ':',
    'pred':     '--',
    'aug':      '-.'
}

PLOT_CONFIG = {
    'figure.figsize': (12, 6),
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'font.size': 12
}

DEFAULT_BATCH_SIZE = 32
MAX_EPOCHS = 150
PATIENCE = 15  # Early Stopping