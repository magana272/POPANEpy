import os


def get_size(start_path='.', units='GB') -> dict[str, dict[str, float | int]]:
    """Get size in GB of csv files per study
    in a given directory    
    """
    size_per_study: dict[str, dict[str, float | int]] = dict({
        "S1": {"GB": 0.0, "File_Count": 0},
        "S2": {"GB": 0.0, "File_Count": 0},
        "S3": {"GB": 0.0, "File_Count": 0},
        "S4": {"GB": 0.0, "File_Count": 0},
        "S5": {"GB": 0.0, "File_Count": 0},
        "S6": {"GB": 0.0, "File_Count": 0},
        "S7": {"GB": 0.0, "File_Count": 0}})
    for dirpath, __, filenames in os.walk(start_path, followlinks=True):
        for f in filenames:
            if ".csv" in f:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    size_per_study[f[:2]][units] += os.path.getsize(fp)
                    size_per_study[f[:2]]["File_Count"] += 1
    for study, values in size_per_study.items():
        if units == 'GB':
            values["GB"] = values["GB"] / (1024**3)
            size_per_study[study] = values
        else:
            raise ValueError("Unsupported unit. Please use 'GB'.")
    return size_per_study
