"""
Converts from LLM's prediction JSON format to TSV format for mt-metrics-eval.
"""

# system	doc	doc_id	seg_id	rater	source	target	category	severity

# Example without error
# Online-B	news_stv.tv.1681:English-German	1	956	rater1	Warning of stormy weather as strong winds present 'danger to life'	Warnung vor stürmischem Wetter, da starke Winde „Lebensgefahr“ darstellen	No-error	No-error

# Example with
# M2M100_1.2B-B4	social_t1_hkc5no1:English-German	1	1026	rater1	Amazing.	<v>erstaunlich ist</v>.	Accuracy/Mistranslation	major


import json
import argparse

def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.splitlines() if line.strip()]
    return data


def convert_to_tsv(input_file, output_tsv):
    data = load_json_or_jsonl(input_file)

    with open(output_tsv, 'w', encoding='utf-8') as f:
        # Write header
        f.write("system\tdoc\tdoc_id\tseg_id\trater\tsource\ttarget\tcategory\tseverity\n")
        
        for entry in data:
            system_id = entry.get("system", "default")
            doc = "moedict_select"
            doc_id = entry.get("id")
            rater_id = entry.get("rater", "")

            source = entry.get("src", "").replace("\t", " ").replace("\n", " ")
            candidate = entry.get("mt", "").replace("\t", " ").replace("\n", " ")

            annotations = entry.get("annotations", entry.get("errors", entry.get("spans", [])))
            seg_id = 0
            
            if not annotations:
                line = f"{system_id}\t{doc}\t{doc_id}\t{seg_id}\t{rater_id}\t{source}\t{candidate}\tno-error\tno-error\n"
                f.write(line)
                continue
   
            for span in annotations:
                category = span.get("error_type", span.get("category", ""))
                severity = span.get("error_severity", span.get("severity", ""))
                
                # Insert start and end index tag to the candidate
                start_idx = span.get("start_index", span.get("start", -1))
                end_idx = span.get("end_index", span.get("end", -1))
                
                
                if start_idx == end_idx == 0 and category in ("No-error/", "No-error/None"):
                    print(f"id:{doc_id}, no-error span")
                    line = f"{system_id}\t{doc}\t{doc_id}\t{seg_id}\t{rater_id}\t{source}\t{candidate}\tno-error\tno-error\n"
                    f.write(line)
                    continue
                elif start_idx == end_idx == 0 and category in ("Non-translated/None", "Non-translated/"):
                    print(f"id:{doc_id}, non-translated span")
                    line = f"{system_id}\t{doc}\t{doc_id}\t{seg_id}\t{rater_id}\t{source}\t{candidate}\tNon-translated\tCritical\n"
                    f.write(line)
                    continue
                # Rule out invalid indices
                elif start_idx < 0 or end_idx < 0 or start_idx >= end_idx or end_idx > len(candidate):
                    print(f"id:{doc_id}, start_idx: {start_idx}, end_idx: {end_idx}, {len(candidate)}")
                    continue
                
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    new_candidate = candidate[:start_idx] + "<v>" + candidate[start_idx:end_idx] + "</v>" + candidate[end_idx:]

                line = f"{system_id}\t{doc}\t{doc_id}\t{seg_id}\t{rater_id}\t{source}\t{new_candidate}\t{category}\t{severity}\n"
                f.write(line)
                
                seg_id += 1
    

    print(f"Converted {input_file} to {output_tsv}")
    with open(output_tsv, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # find the number of different doc_id
        doc_ids = set()
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) > 2:
                doc_ids.add(parts[2])
        print(f"Number of unique doc_id: {len(doc_ids)}")

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Convert JSON/JSONL to TSV for mt-metrics-eval")
    ap.add_argument("--model_name", type=str, required=True, default="gpt-4o", help="Model name for input/output file naming")
    args = ap.parse_args()

    MODEL_NAME = args.model_name
    INPUT_JSON_FILE = f"./pred_output/{MODEL_NAME}.jsonl"
    convert_to_tsv(INPUT_JSON_FILE, f"./model_tsv/{MODEL_NAME}.tsv")