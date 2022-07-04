import pandas as pd
import os
import argparse


my_parser = argparse.ArgumentParser(description='Get the top-1 prompt for LPAQA')
my_parser.add_argument("--relation-dir", help="relations directory", type=str, default="data/TREx/")
my_parser.add_argument("--relation-file", help="relation file", type=str)
my_args = my_parser.parse_args()

dir = my_args.relation_dir
relations = []
for file in os.listdir(dir):
        relations.append(pd.read_json(os.path.join(dir,file), lines=True))

        combined_relation = pd.concat(relations)
        combined_relation = combined_relation.reset_index()
        combined_relation = combined_relation.sort_values(["relation","index"]).reset_index(drop=True)

df = pd.read_csv(my_args.relation_file, header=None).rename(columns={0:"relation_1",1:"score"})
df = df.reset_index()
df = df.sort_values(["relation_1", "index"]).reset_index(drop=True)
result = pd.concat([combined_relation, df], axis=1)
max_index = result.groupby("relation")["score"].idxmax()

relation_final = result.iloc[max_index][["relation","template"]]
output_file = "top_1_lpaqa_" + my_args.relation_file.rsplit(".",1)[0] + ".jsonl"
with open(output_file, "w") as outfile:
        outfile.write(relation_final.to_json(orient="records", lines=True))