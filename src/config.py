DEFAULT_PATH = "C:/Users/oldan/Desktop/RuralDevelopment/progetto-tirocinio/data/processed_italy_data.xlsx"
RESULTS_PATH = "C:/Users/oldan/Desktop/RuralDevelopment/progetto-tirocinio/results/models_performance.xlsx"

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
    'train': 'black',
    'test': '#1f77b4',
    'pred': '#d62728',
    'pred_multi': ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
}