import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Expanded Region 7 Configuration
NUM_RECORDS = 5000
LOCATIONS = [
    # Cebu Province - South
    "Jampang, Argao, Cebu", 
    "Tulic, Argao, Cebu",
    "Mantalongon, Dalaguete, Cebu", 
    "Obo, Dalaguete, Cebu",
    "Campesa, Barili, Cebu", 
    "Guadalupe, Carcar, Cebu", 
    "Lamacan, Sibonga, Cebu",
    # Cebu Province - Central / City Uplands
    "San Nicholas, Cebu City, Cebu",
    "Sudlon II, Cebu City, Cebu",
    "Guba, Cebu City, Cebu",
    # Cebu Province - West / North
    "Cansomoroy, Balamban, Cebu", 
    "Luray, Toledo, Cebu",
    "Gairan, Bogo City, Cebu",
    # Bohol Province
    "Buenos Aires, Carmen, Bohol", 
    "Nueva Fuerza, Carmen, Bohol",
    "Biabas, Ubay, Bohol", 
    "San Pascual, Ubay, Bohol",
    # Negros Oriental Province
    "Pula, Canlaon, Negros Oriental", 
    "Linothangan, Canlaon, Negros Oriental"
]

CROPS = ["Sorghum", "Corn", "Rice", "Cabbage", "Tomato", "Sugarcane"]
APP1_BLENDS = ["143 kg/ha (14-14-14)", "100 kg/ha (14-14-14)", "150 kg/ha (16-20-0)", "200 kg/ha (14-14-14)"]
APP2_BLENDS = ["24 kg/ha (21-0-0)", "50 kg/ha (46-0-0)", "30 kg/ha (46-0-0)", "None"]

def generate_synthetic_soil_data(num_records):
    data = []
    start_date = datetime(2025, 1, 1)
    
    for i in range(num_records):
        # Metadata
        lab_code = f"SCC-26-{400 + i}"
        site = random.choice(LOCATIONS)
        date_sampled = start_date + timedelta(days=random.randint(0, 500))
        crop = random.choice(CROPS)
        farm_area = round(random.uniform(100, 10000), 2) # Sq. M
        
        # Input Features (Numerical)
        ph = round(random.uniform(4.5, 8.5), 2)
        om = round(random.uniform(0.5, 5.0), 2)
        phosphorus = round(random.uniform(5.0, 40.0), 2)
        potassium = round(random.uniform(50.0, 250.0), 2)
        nitrogen = round(om * 0.05, 4) 
        
        # Qualitative Remarks (Categorical logic)
        ph_remark = "MODERATELY ALKALINE" if ph > 7.5 else ("ACIDIC" if ph < 6.0 else "NEUTRAL")
        om_remark = "HIGH" if om > 3.0 else ("MODERATELY LOW" if om > 1.5 else "LOW")
        p_remark = "HIGH" if phosphorus > 20 else "LOW"
        k_remark = "SUFFICIENT+" if potassium > 100 else "DEFICIENT"
        
        # Targets (Simulated Logic)
        req_n = round(max(20, 100 - (nitrogen * 1000)), 0)
        req_p = round(max(20, 60 - phosphorus), 0)
        req_k = round(max(20, 80 - (potassium * 0.2)), 0)
        
        app1 = random.choice(APP1_BLENDS)
        app2 = random.choice(APP2_BLENDS)
        
        data.append([
            lab_code, site, date_sampled.strftime('%Y-%m-%d'), crop, farm_area,
            ph, om, phosphorus, potassium, nitrogen,
            ph_remark, om_remark, p_remark, k_remark,
            req_n, req_p, req_k, app1, app2
        ])
        
    columns = [
        "laboratory_code", "sampling_site", "date_of_sampling", "crop", "farm_area_sqm",
        "ph", "organic_matter_pct", "phosphorus_ppm", "potassium_ppm", "total_nitrogen_pct",
        "ph_remark", "om_remark", "p_remark", "k_remark",
        "target_req_n", "target_req_p2o5", "target_req_k2o",
        "app1_formulation", "app2_formulation"
    ]
    
    return pd.DataFrame(data, columns=columns)

# Generate and save to CSV
df_synthetic = generate_synthetic_soil_data(NUM_RECORDS)
df_synthetic.to_csv("suri_synthetic_training_data_region7.csv", index=False)
print(f"Successfully generated {NUM_RECORDS} synthetic records spanning DA Region 7.")