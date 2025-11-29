import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util


def extract_data_from_logs(logs_dir):
    """
    Extract data from all {subject}_wrongs.csv files in the logs directory.

    Args:
        logs_dir: Path to the logs directory

    Returns:
        List of dictionaries containing extracted data with subject names
    """
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    # Find all *_wrongs.csv files
    csv_files = list(logs_path.glob("*_wrongs.csv"))

    if not csv_files:
        print(f"Warning: No *_wrongs.csv files found in {logs_dir}")
        return []

    all_data = []

    for csv_file in csv_files:
        # Extract subject name from filename (e.g., "과학" from "과학_wrongs.csv")
        subject = csv_file.stem.replace("_wrongs", "")

        try:
            df = pd.read_csv(csv_file, encoding="utf-8")

            # Validate required columns
            required_columns = [
                "true_code",
                "pred_code",
                "match_type",
                "true_content",
                "pred_content",
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: {csv_file.name} is missing columns: {missing_columns}")
                continue

            # Add subject column and convert to list of dicts
            for _, row in df.iterrows():
                all_data.append(
                    {
                        "subject": subject,
                        "true_code": row["true_code"],
                        "pred_code": row["pred_code"],
                        "match_type": row["match_type"],
                        "true_content": (
                            str(row["true_content"])
                            if pd.notna(row["true_content"])
                            else ""
                        ),
                        "pred_content": (
                            str(row["pred_content"])
                            if pd.notna(row["pred_content"])
                            else ""
                        ),
                    }
                )

            print(f"Extracted {len(df)} rows from {csv_file.name}")

        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue

    return all_data


def calculate_cosine_similarities(data, model_name="jhgan/ko-sroberta-multitask"):
    """
    Calculate cosine similarity between true_content and pred_content using SentenceTransformer.
    For invalid entries, set cos_sim to -1.

    Args:
        data: List of dictionaries with true_content and pred_content
        model_name: SentenceTransformer model name for semantic similarity

    Returns:
        List of dictionaries with added cos_sim field
    """
    results = []

    # Separate invalid and valid entries
    invalid_entries = []
    valid_entries = []

    for entry in data:
        if entry["match_type"] == "invalid":
            invalid_entries.append(entry)
        else:
            valid_entries.append(entry)

    # Process invalid entries (set cos_sim to -1)
    for entry in invalid_entries:
        entry["cos_sim"] = -1
        results.append(entry)

    # Process valid entries
    if valid_entries:
        # Prepare texts
        true_texts = [entry["true_content"] for entry in valid_entries]
        pred_texts = [entry["pred_content"] for entry in valid_entries]

        # Handle empty strings
        true_texts = [text if text and text.strip() else " " for text in true_texts]
        pred_texts = [text if text and text.strip() else " " for text in pred_texts]

        # Calculate SentenceTransformer semantic similarity
        print("  Calculating SentenceTransformer semantic similarities...")
        try:
            model = SentenceTransformer(model_name)
            model.eval()

            # Encode texts in batches for efficiency
            batch_size = 32
            cos_sim_list = []

            for i in range(0, len(true_texts), batch_size):
                batch_true = true_texts[i : i + batch_size]
                batch_pred = pred_texts[i : i + batch_size]

                # Encode batches
                true_embeddings = model.encode(
                    batch_true, convert_to_tensor=True, show_progress_bar=False
                )
                pred_embeddings = model.encode(
                    batch_pred, convert_to_tensor=True, show_progress_bar=False
                )

                # Calculate cosine similarity for each pair in the batch
                for j in range(len(batch_true)):
                    cos_sim = util.cos_sim(
                        true_embeddings[j : j + 1], pred_embeddings[j : j + 1]
                    ).item()
                    cos_sim_list.append(float(cos_sim))

            # Assign semantic similarities to entries
            for i, entry in enumerate(valid_entries):
                entry["cos_sim"] = cos_sim_list[i]
                results.append(entry)

        except Exception as e:
            print(f"Error calculating SentenceTransformer cosine similarity: {e}")
            print(
                f"  Install sentence-transformers if needed: pip install sentence-transformers"
            )
            # Fallback: set cos_sim to 0 for valid entries if calculation fails
            for entry in valid_entries:
                entry["cos_sim"] = 0.0
                results.append(entry)

    return results


def create_analysis_csv(data, output_path):
    """
    Create error_analysis.csv with columns: no, subject, match_type, cos_sim, true_content, pred_content

    Args:
        data: List of dictionaries with processed data
        output_path: Path to save the CSV file
    """
    if not data:
        print("Warning: No data to write to CSV")
        return

    # Create DataFrame
    df = pd.DataFrame(data)

    # Select and reorder columns
    df_output = pd.DataFrame(
        {
            "no": range(1, len(df) + 1),
            "subject": df["subject"],
            "match_type": df["match_type"],
            "cos_sim": df["cos_sim"],
            "true_content": df["true_content"],
            "pred_content": df["pred_content"],
        }
    )

    # Save to CSV
    df_output.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Created error_analysis.csv with {len(df_output)} rows at {output_path}")


def create_summary_csv(data, output_path):
    """
    Create error_summary.csv with subject-wise statistics for cosine similarity.

    Args:
        data: List of dictionaries with processed data
        output_path: Path to save the CSV file
    """
    if not data:
        print("Warning: No data to write to CSV")
        return

    # Group by subject
    summary = {}

    # Initialize summary with 0.1 unit ranges from 0 to 1.0
    for entry in data:
        subject = entry["subject"]
        if subject not in summary:
            summary[subject] = {
                "invalid": 0,
                "cossim_0_0.1": 0,
                "cossim_0.1_0.2": 0,
                "cossim_0.2_0.3": 0,
                "cossim_0.3_0.4": 0,
                "cossim_0.4_0.5": 0,
                "cossim_0.5_0.6": 0,
                "cossim_0.6_0.7": 0,
                "cossim_0.7_0.8": 0,
                "cossim_0.8_0.9": 0,
                "cossim_0.9_1.0": 0,
            }

        cos_sim = entry.get("cos_sim", -1)

        # Categorize cosine similarity
        # Note: SentenceTransformer cosine similarity is always in range [0, 1]
        # -1 is used as a special flag for invalid entries
        if cos_sim == -1:
            summary[subject]["invalid"] += 1
        elif 0 <= cos_sim < 0.1:
            summary[subject]["cossim_0_0.1"] += 1
        elif 0.1 <= cos_sim < 0.2:
            summary[subject]["cossim_0.1_0.2"] += 1
        elif 0.2 <= cos_sim < 0.3:
            summary[subject]["cossim_0.2_0.3"] += 1
        elif 0.3 <= cos_sim < 0.4:
            summary[subject]["cossim_0.3_0.4"] += 1
        elif 0.4 <= cos_sim < 0.5:
            summary[subject]["cossim_0.4_0.5"] += 1
        elif 0.5 <= cos_sim < 0.6:
            summary[subject]["cossim_0.5_0.6"] += 1
        elif 0.6 <= cos_sim < 0.7:
            summary[subject]["cossim_0.6_0.7"] += 1
        elif 0.7 <= cos_sim < 0.8:
            summary[subject]["cossim_0.7_0.8"] += 1
        elif 0.8 <= cos_sim < 0.9:
            summary[subject]["cossim_0.8_0.9"] += 1
        elif 0.9 <= cos_sim <= 1.0:
            summary[subject]["cossim_0.9_1.0"] += 1
        else:
            print(f"Warning: Unexpected cos_sim value {cos_sim} for subject {subject}")

    # Convert to DataFrame and save as CSV
    summary_list = []
    for subject, counts in summary.items():
        row = {"subject": subject}
        row.update(counts)
        summary_list.append(row)

    df_summary = pd.DataFrame(summary_list)

    # Sort by specified subject order
    subject_order = [
        "과학",
        "국어",
        "기술가정",
        "도덕",
        "사회",
        "사회문화",
        "수학",
        "영어",
        "정보",
    ]
    # Create a mapping for sorting
    subject_order_dict = {subject: idx for idx, subject in enumerate(subject_order)}
    # Add subjects not in the order list to the end
    max_order = len(subject_order)
    df_summary["_sort_order"] = df_summary["subject"].map(
        lambda x: subject_order_dict.get(x, max_order + hash(x))
    )
    df_summary = df_summary.sort_values("_sort_order").drop("_sort_order", axis=1)

    df_summary.to_csv(output_path, index=False, encoding="utf-8")

    print(
        f"Created error_summary.csv with statistics for {len(summary)} subjects at {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze error patterns from RAG LLM text classification results"
    )
    parser.add_argument(
        "logs_dir",
        type=str,
        help="Path to the logs directory containing *_wrongs.csv files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="jhgan/ko-sroberta-multitask",
        help="SentenceTransformer model name for semantic similarity (default: jhgan/ko-sroberta-multitask)",
    )

    args = parser.parse_args()

    # Extract data from CSV files
    print(f"Extracting data from {args.logs_dir}...")
    data = extract_data_from_logs(args.logs_dir)

    if not data:
        print("No data extracted. Exiting.")
        return

    print(f"Total rows extracted: {len(data)}")

    # Calculate cosine similarities
    print("Calculating cosine similarities...")
    print(f"  Using SentenceTransformer model: {args.model_name}")
    data_with_cossim = calculate_cosine_similarities(data, model_name=args.model_name)

    # Use logs_dir as output directory
    output_dir = Path(args.logs_dir)

    # Create error_analysis.csv
    analysis_csv_path = output_dir / "error_analysis.csv"
    create_analysis_csv(data_with_cossim, analysis_csv_path)

    # Create error_summary.csv
    summary_csv_path = output_dir / "error_summary.csv"
    create_summary_csv(data_with_cossim, summary_csv_path)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
