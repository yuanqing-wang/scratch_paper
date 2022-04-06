import json
## Load in data from output JSON file:
path = "output.json" # path to the JSON file
with open(path) as path_file:
    data = json.load(path_file)

import pandas as pd
df = pd.DataFrame(data["pathwayProfile"])
df.to_csv("output.csv")
