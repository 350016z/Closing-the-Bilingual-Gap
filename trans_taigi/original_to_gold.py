import json 

with open("moedict_select-taigi.json", "r", encoding="utf-8") as f:
    data = json.load(f)

output = []
for key, item in data.items():

    # --- Remove duplicate text within spans ---
    if "spans" in item and isinstance(item["spans"], list):
        unique_spans = []
        seen_texts = set()
        for span in item["spans"]:
            text = span.get("text", "")
            if text not in seen_texts:
                unique_spans.append(span)
                seen_texts.add(text)
        item["spans"] = unique_spans

    # --- Rename fields ---
    if "source" in item:
        item["src"] = item.pop("source")
    if "target" in item:
        item["mt"] = item.pop("target")
    if "reference" in item:
        item["ref"] = item.pop("reference")
    if "system" in item:
        item.pop("system")  # Remove the system field

    # --- Adjust severity for Non-translated/None ---
    for span in item.get("spans", []):
        if span.get("category") in ("Non-translated/None", "Non-translated/"):
            span["severity"] = "Critical"


    output.append(item)



# Select records whose category == "Non-translated/None"
non_translated = []
for item in output:
    for span in item.get("spans", []):
        if span.get("category") == "Non-translated/None":
            non_translated.append(item["id"])
            break  # Count a record once if any of its spans matches

# Counts
total = len(output)
non_translated_count = len(non_translated)
remaining = total - non_translated_count

# Output results
print(f"Total records: {total}")
print(f"Records with category 'Non-translated/None': {non_translated_count}")
print("Their ids:", non_translated)

# Drop the "Non-translated/None" records
# filtered_data = [item for item in output if any(span.get("category") != "Non-translated/None" for span in item.get("spans", []))]
with open("gold.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)




print("✅ 轉換完成，結果已輸出至 gold.json")
