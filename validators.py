def normalize_lookup(value: str):
    return str(value).strip().casefold()

def validate_job(job: str, job_list):
    if not job:
        return None
    targets = {normalize_lookup(x): x for x in job_list}
    return targets.get(normalize_lookup(job))

def validate_region(region: str, region_list):
    if not region:
        return None
    targets = {normalize_lookup(x): x for x in region_list}
    return targets.get(normalize_lookup(region))

def validate_income(x):
    try:
        v = float(x)
        return v if v >= 1000 else None
    except Exception:
        return None

def validate_dependents(x):
    try:
        v = int(float(x))
        return v if v >= 0 else None
    except Exception:
        return None

def missing_fields(profile: dict):
    missing = []
    if not profile.get("Job"): missing.append("job")
    if not profile.get("Region"): missing.append("region")
    if profile.get("Monthly_Income") is None: missing.append("income")
    if profile.get("Number_of_Dependents") is None: missing.append("dependents")
    return missing
