import sys
import re
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.preprocessor import extract_numeric_value

def parse_param_cell(txt_key, val_from_sheet):
    t_lower = txt_key.lower()
    
    # 1. Identify Parameter Name
    p_name = None
    p_unit = ""
    if "interface defect density 1" in t_lower or "idd1" in t_lower:
        p_name, p_unit = "Interface_Defect_Density_1", "cm^-3"
    elif "interface defect density 2" in t_lower or "idd2" in t_lower:
        p_name, p_unit = "Interface_Defect_Density_2", "cm^-3"
    elif "defect density" in t_lower or "total defect density" in t_lower or "defect 1" in t_lower or "idd" in t_lower:
        p_name, p_unit = "Defect_Density", "cm^-3"
    elif "thickness" in t_lower or "thick" in t_lower or "t[" in t_lower or "t(" in t_lower:
        p_name, p_unit = "Thickness", "um"
    elif "temperature" in t_lower or "temp" in t_lower or "t(k)" in t_lower:
        p_name, p_unit = "Temperature", "K"
        
    if not p_name:
        return None
        
    # 2. Extract Value: Check text after ':' or '=' first
    num_val = None
    if ":" in txt_key or "=" in txt_key:
        rhs = re.split(r"[:=]", txt_key)[-1].strip()
        num_val = extract_numeric_value(rhs)
        
    if num_val is None:
        num_val = val_from_sheet
        
    return p_name, p_unit, num_val

print("Testing 'Ca3PI3 (L3)>>thickness[µm]:' with cell val 0.3:")
res1 = parse_param_cell("Ca3PI3 (L3)>>thickness[µm]:", 0.3)
print(f"Result 1: {res1}")

print("\nTesting 'Ca3PI3 (L3)>>defect 1>>total defect density [1/cm³]:' with cell val 1e11:")
res2 = parse_param_cell("Ca3PI3 (L3)>>defect 1>>total defect density [1/cm³]:", 100000000000.0)
print(f"Result 2: {res2}")
