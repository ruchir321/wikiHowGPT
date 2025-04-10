import pandas as pd
import json

# chunksize = 1000
inp = "data/wikihowAll.csv"
# out = "data/wikihowAll.json"
# out = "data/wikihow_100.json"
out = "data/wikihow_10.json"

with open(out, "w", encoding="utf-8") as json_file:
    # json_file.write("[\n")

    # first_chunk = True

    df = pd.read_csv(inp, nrows=10)

    records = df.to_dict(orient="records")

    json.dump(obj=records, fp=json_file, ensure_ascii=False, indent=4)

    # for chunk in pd.read_csv(inp, chunksize=chunksize):
    #     chunk.dropna(inplace=True)
    #     records = chunk.to_dict(orient="records")

    #     if not first_chunk:
    #         json_file.write(",\n")
        
    #     else:
    #         first_chunk = False
        
    #     json.dump(records, json_file, ensure_ascii=False, indent=4)
    
    # json_file.write("\n]")
