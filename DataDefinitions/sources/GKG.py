import pandas as pd
import requests
from credentials import GOOGLE_API_KEY as API_KEY
import urllib.parse
import urllib.request
import json
import time

def get_knowledge_graph_terms(company_name):
    service_url = "https://kgsearch.googleapis.com/v1/entities:search"
    params = {
        "query": company_name,
        "limit": 10,  # Set the limit to 20 to match the example
        "indent": True,
        "key": API_KEY
    }
    url = service_url + '?' + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read())
    entity_scores = []
    for element in data["itemListElement"]:
        result = element.get("result", {})
        entity_name = result.get("name", "")
        entity_score = element.get("resultScore", 0)
        if entity_name:
            entity_scores.append((entity_name, entity_score))
    entity_scores.sort(key=lambda x: x[1], reverse=True)

    return [entity[0] for entity in entity_scores[:20]]






all_data = pd.read_csv('/Users/lukavulicevic/Desktop/LLM Lookahead Bias/Analysis Ready/Merged/final_output_all_articles_CRSP_COMP_MORE_NITY.csv', low_memory=False)

unique_comnam = all_data["COMNAM"].unique()
GKG = pd.DataFrame(index=unique_comnam, columns=["GKG"])

total = len(GKG)
for i, company in enumerate(GKG.index):
    result = get_knowledge_graph_terms(company)
    GKG.at[company, "GKG"] = result
    print(f"\rProcessing; [{(i + 1) / total * 100:.2f}% complete]", end="", flush=True)


GKG.index.name = 'COMNAM'

all_data = all_data.merge(GKG, on='COMNAM', how='left')
#all_data.drop(columns='GKG',inplace=True)

def preprocess_gkg(gkg_string):
    stop_words = {"the", "and", "of", "in", "to", "for", "on", "with", "at", "by", "from", "as", "an", "is", "that",
                  "it", "this", "be", "are", "was", "were", "has", "had", "will", "would", "can", "could", "should",
                  "a"}
    try:
        words = set()
        gkg_list = ast.literal_eval(gkg_string) if isinstance(gkg_string, str) else gkg_string
        if isinstance(gkg_list, list):
            for phrase in gkg_list:
                for word in phrase.split():
                    if word.lower() not in stop_words:
                        words.add(word)
        return list(words)
    except (SyntaxError, ValueError):
        return gkg_string

all_data["GKG"] = all_data["GKG"].apply(preprocess_gkg)


all_data["GKG_MASKED"] = all_data.apply(GKG_mask, axis=1)
all_data["FIRM_MASKED"] = all_data.apply(mask_text, axis=1)

all_data.head(10).to_clipboard()


all_data.to_csv('/Users/lukavulicevic/Desktop/LLM Lookahead_Bias/Analysis Ready/Merged/final_output_all_articles_CCMore_NITY_MASKS.csv')

