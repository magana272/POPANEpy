import os


def get_size(start_path="./data/raw/", units='GB') -> dict[str, dict[str, float | int]]:
    """Get size in GB of csv files per study
    in a given directory    
    """
    size_per_study: dict[str, dict[str, float | int]] = dict({
        "study1": {"Size": 0.0, "File_Count": 0},
        "study2": {"Size": 0.0, "File_Count": 0},
        "study3": {"Size": 0.0, "File_Count": 0},
        "study4": {"Size": 0.0, "File_Count": 0},
        "study5": {"Size": 0.0, "File_Count": 0},
        "study6": {"Size": 0.0, "File_Count": 0},
        "study7": {"Size": 0.0, "File_Count": 0}})
    # TODO: need to not loop over dir like this
    for dirpath, dirnames, filenames in os.walk(start_path, followlinks=True):
        for dirname in dirnames:
            if "study" in dirname:
                count, size = get_count_and_size(os.path.join(start_path, dirname))
                size_per_study[dirname]["File_Count"] += count
                size_per_study[dirname]["Size"] += size
        break
    for study, values in size_per_study.items():
        if units == 'GB':
            values["Size"] = values["Size"] / (1024 ** 3)
            size_per_study[study] = values
        else:
            raise ValueError("Unsupported unit. Please use 'GB'.")
    return size_per_study


def get_count_and_size(path):
    count = 0
    size = 0
    for dirpath, __, filenames in os.walk(path, followlinks=True):
        for f in filenames:
            if ".csv" in f:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    size += os.path.getsize(fp)
                    count += 1
    return count, size


def main():
    print(get_size('C:\\Users\\super_grool\\Desktop\\POPANEpy\\data\\raw', 'GB'))


if __name__ == "__main__":
    main()
