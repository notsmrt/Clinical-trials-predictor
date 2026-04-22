import pandas as pd

def merge_and_deduplicate_csvs(file1_path, file2_path, output_path):
    df1 = pd.read_csv(file1_path)

    df2 = pd.read_csv(file2_path, header=None)
    df2.columns = df1.columns

    # concatenate the two dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # remove duplicate rows
    deduplicated_df = combined_df.drop_duplicates()

    # save the result to a new CSV
    deduplicated_df.to_csv(output_path, index=False)
    print(f"Merged and deduplicated data saved to {output_path}")

if __name__ == "__main__":
    file_a = "merged-studies.csv"
    file_b = "ctg1-studies.csv"
    output_file = "merged-studies.csv"

    merge_and_deduplicate_csvs(file_a, file_b, output_file)
