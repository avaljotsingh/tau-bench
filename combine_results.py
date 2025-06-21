import os
import json

input_folder = '.'
output_file = 'assertions-agent-none-0.1_range_0-100_6_20_2025_without_postcondition.json'

combined = []
files = ['assertions-agent-none-0.1_range_0-21_user-none-llm_0620105151.json', 'assertions-agent-none-0.1_range_21-100_user-none-llm_0619144914.json', 'assertions-agent-none-0.1_range_74-100_user-none-llm_0620100211.json']

for filename in os.listdir(input_folder):
    if filename in files:
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    print(len(data), file_path)
                    combined.extend(data)
                else:
                    print(f"⚠️ Skipping {filename} — not a list at top level.")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to load {filename}: {e}")


# print(len(combined))
# Write the combined list to the output file
with open(output_file, 'w') as f:
    json.dump(combined, f, indent=2)

print(f"✅ Combined {len(combined)} items into: {output_file}")
