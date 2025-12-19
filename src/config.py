import tensorflow as tf
import numpy as np
import random
import os

# path directory del progetto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# path directory specifiche di salvataggio risultati
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
ERRORS_DIR = os.path.join(RESULTS_DIR, "errors")
MODELS_DIR = os.path.join(RESULTS_DIR, "saved_models")

# path file specifici da aggiornare durante esecuzione modelli
RAW_DATA_FILE = os.path.join(DATA_DIR, "processed_italy_data.xlsx")
PERFORMANCE_FILE = os.path.join(ERRORS_DIR, "performance_metrics.xlsx")
PARAMS_FILE = os.path.join(LOGS_DIR, "model_params.xlsx")
RESIDUALS_FILE = os.path.join(ERRORS_DIR, "residuals_diagnostics.xlsx")
PREDICTIONS_FILE = os.path.join(LOGS_DIR, "future_forecasts.csv")

# nomi indicatori inusabili
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

# mappa di rinomina indicator name -> long : short 
SAFE_VAR_NAMES = {
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

# mappa di rinomina inversa indicator name -> short : long  (per plotting)
REVERSE_VAR_NAMES = {v: k for k, v in SAFE_VAR_NAMES.items()}

def get_complete_path(base_dir, model_name, variable_name, ext=".png"):
    """
    Genera il percorso completo per il salvataggio e crea la cartella se necessario.
    
    Args:
        base_dir (str): La cartella radice (es. PLOTS_DIR o MODELS_DIR).
        model_name (str): Il nome del modello (es. 'CNN_Shallow').
        variable_name (str): Il nome dell'indicatore
        ext (str): L'estensione del file (default '.png', usa '.keras' per modelli).
        
    Returns:
        str: Il percorso assoluto completo del file.
    """
    save_folder = os.path.join(base_dir, str(model_name))
    os.makedirs(save_folder, exist_ok=True) 
    
    safe_var = SAFE_VAR_NAMES.get(variable_name, variable_name[:3])
    safe_var = "".join([c if c.isalnum() else "_" for c in safe_var])[:50]
    
    filename = f"{str(model_name)}_{safe_var}{ext}"
    return os.path.join(save_folder, filename)

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
RANDOM_SEED = 42

def set_seeds(seed=RANDOM_SEED):
    """ Sets random seeds for Python, NumPy, and TensorFlow to ensure
    reproducible results across execution runs.

    This function also attempts to enable deterministic operations in TensorFlow,
    which is crucial for reproducibility on GPUs, though it might slightly
    impact performance.

    Args:
        seed (int, optional): The seed value to be used. Defaults to RANDOM_SEED.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Determinismo operazioni GPU/CPU (TensorFlow >= 2.9)
    # Questa funzione rende deterministici anche i layer convoluzionali
    # che solitamente introducono rumore stocastico su GPU.
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        # Fallback per versioni vecchie di TF
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        
    print(f"Random seeds set to {seed}. Deterministic operations enabled.")