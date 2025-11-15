# ----------------------------------------------------------------------
# INSTRUCTIONS:
# 1. Ensure you have run: pip install pandas
# 2. Place this file ('medquad_converter.py') in the same folder as 'medquad.csv'.
# 3. Run the script: python medquad_converter.py
# ----------------------------------------------------------------------

import pandas as pd
import json

# --- CONFIGURATION ---
INPUT_CSV_FILE = 'medquad.csv' 
OUTPUT_JSONL_FILE = 'health_myth_data.jsonl' 

# Map the columns in medquad.csv to the required categories:
# 'question' -> Claim, 'answer' -> Analysis
COLUMN_MAPPING = {
    'claim_column': 'question',
    'explanation_column': 'answer'
}

# --- CONVERSION LOGIC ---

def convert_medquad_to_jsonl(input_file, output_file, mapping):
    """Reads MedQuAD CSV and formats it into the Instruction-Tuning JSONL format."""
    
    try:
        # Load the dataset
        # We assume the first row contains headers
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"ERROR: Input file '{input_file}' not found. Please check the file name.")
        return
    
    print(f"Successfully loaded {len(df)} rows from {input_file}.")
    
    processed_count = 0
    # Use 'w' (write) mode to START a new file. If you have other data, change to 'a' (append).
    with open(output_file, 'w', encoding='utf-8') as f:
        
        # Iterate through the rows of the DataFrame
        for index, row in df.iterrows():
            try:
                # 1. Extract raw data
                claim = str(row[mapping['claim_column']]).strip()
                explanation = str(row[mapping['explanation_column']]).strip()
                
                # Simple cleanup: replace internal newlines and multiple spaces with a single space
                explanation = ' '.join(explanation.split())

                # 2. Assemble the full factual response string
                # Since MedQuAD is verified Q&A data, we assign a "Fact" verdict.
                response_content = (
                    f"**Verdict: Fact**\n"
                    f"**Analysis:** {explanation}\n"
                    f"**Recommendation:** Consult a doctor for any symptoms or medical concerns."
                )
                
                # 3. Assemble the full training instruction
                instruction = (
                    "## Health Claim Checker\n\n"
                    f"### Claim\n{claim}\n\n"
                    f"### Response\n{response_content}"
                )
                
                # 4. Write to JSONL file
                json_line = json.dumps({"text": instruction}, ensure_ascii=False)
                f.write(json_line + '\n')
                processed_count += 1
                
            except KeyError as e:
                print(f"Skipping row {index}: Column key error: {e}. Check the column mapping.")
                # We stop on a KeyError because the column mapping is fundamentally wrong
                break
            except Exception as e:
                # Skip rows with unexpected errors (e.g., missing data)
                print(f"Skipping row {index} due to unexpected error: {e}")
                
    print(f"\n--- Conversion Complete ---\nSuccessfully processed {processed_count} entries into {output_file}.")
    print("\nIMPORTANT: Next, you MUST manually add 10-20 'Non-Health Rejection' examples to this file!")

# Execute the conversion
if __name__ == "__main__":
    convert_medquad_to_jsonl(INPUT_CSV_FILE, OUTPUT_JSONL_FILE, COLUMN_MAPPING)