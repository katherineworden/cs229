def extract_requests(input_csv_path, output_csv_path):
    import csv
    
    with open(input_csv_path, mode="r", encoding="utf-8") as infile, \
         open(output_csv_path, mode="w", encoding="utf-8", newline="") as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Read and rewrite the header (assumes "goal,target" format)
        header = next(reader, None)
        writer.writerow([header[0]])  # Keep only the first column header
        
        # For each line, keep only the first column (the user request)
        for row in reader:
            writer.writerow([row[0]]) 


if __name__ == "__main__":
    import sys
    
    # Usage: python extract_requests.py input.csv output.csv
    if len(sys.argv) < 3:
        print("Usage: python extract_requests.py <input_csv_path> <output_csv_path>")
    else:
        extract_requests(sys.argv[1], sys.argv[2]) 