import argparse
import os
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Simplified type definitions
FILTER_TYPE = ["no-filter", "exact", "partial"]
PROMPT_TYPE = ["naive", "simple", "verbose", "instruct"]
SECTION_TYPE = ["findings", "impression", "both"]


def generate_from_labels_only(
        model: str,
        reports_csv: str,
        labels_csv: str,
        k: int = 5,
        filter_type: str = "no-filter",
        prompt_type: str = "simple",
        section_type: str = "findings",
        batch_size: int = 32,
        prompt_yaml: str = "prompts_en.yaml",
        output_dir: str = "./output",
        study_id_col: str = "study_id",
        findings_col: str = "findings",
        impression_col: str = "impression",
):
    """
    Minimal report generator: Only requires reports file and labels file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    model_name = os.path.basename(model)
    filename = f"generated_{section_type}_k{k}_{filter_type}_{prompt_type}_{model_name}.csv"
    result_csv = os.path.join(output_dir, filename)

    if os.path.exists(result_csv):
        print(f"Output file exists: {result_csv}")
        return

    print("Loading data...")
    # Load reports data
    reports_df = pd.read_csv(reports_csv)
    # Load labels data
    labels_df = pd.read_csv(labels_csv)

    # Merge reports and labels
    merged_df = reports_df.merge(labels_df, on=study_id_col, how='inner')

    # Get label column names (exclude study_id and other metadata columns)
    label_cols = [col for col in labels_df.columns if col != study_id_col]
    print(f"Found {len(label_cols)} labels: {label_cols}")

    # Filter data with target section
    if section_type == "findings":
        mask = merged_df[findings_col].notna()
    elif section_type == "impression":
        mask = merged_df[impression_col].notna()
    elif section_type == "both":
        mask = merged_df[findings_col].notna() & merged_df[impression_col].notna()
    else:
        raise ValueError(f"Unsupported section type: {section_type}")

    filtered_df = merged_df[mask].copy()
    print(f"Filtered data count: {len(filtered_df)}")

    # Add Other label (if no positive labels)
    other_col = "Other"
    filtered_df[other_col] = (filtered_df[label_cols] != 1).all(axis=1).astype(int)
    all_labels = label_cols + [other_col]

    # Load prompt templates
    with open(prompt_yaml, 'r') as f:
        prompt_templates = yaml.safe_load(f)

    # Setup LLM
    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        use_beam_search=False,
        max_tokens=512,
    )
    llm = LLM(
        model=model,
        dtype='float16',
        trust_remote_code=True,
        enforce_eager=True,
    )

    print("Starting report generation...")
    results = []

    # Generate report for each sample
    for i in tqdm(range(len(filtered_df))):
        current_sample = filtered_df.iloc[i]

        # Prepare retrieval samples (exclude current sample)
        retrieval_samples = filtered_df.drop(i).reset_index(drop=True)

        # Generate pseudo similarity (based on label matching)
        similarity = generate_label_similarity(current_sample, retrieval_samples, all_labels)

        # Prepare prompt
        prompt, target_report, retrieved_studies = prepare_minimal_prompt(
            retrieval_samples=retrieval_samples,
            target_sample=current_sample,
            target_similarity=similarity,
            k=k,
            prompt_templates=prompt_templates,
            filter_type=filter_type,
            prompt_type=prompt_type,
            section_type=section_type,
            labels=all_labels,
            findings_col=findings_col,
            impression_col=impression_col,
            study_id_col=study_id_col,
        )

        # Generate report
        try:
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
        except Exception as e:
            print(f"Generation failed: {e}")
            generated_text = ""

        # Save results
        results.append({
            study_id_col: current_sample[study_id_col],
            "retrieved_studies": retrieved_studies,
            "prompt": prompt,
            "actual_text": target_report,
            "generated_text": generated_text,
        })

        # Save every batch_size samples
        if len(results) >= batch_size:
            save_results(results, result_csv, header=not os.path.exists(result_csv))
            results = []

    # Save remaining results
    if results:
        save_results(results, result_csv, header=not os.path.exists(result_csv))

    print(f"Generation completed! Results saved to: {result_csv}")


def generate_label_similarity(target_sample, retrieval_samples, labels):
    """Generate similarity based on label overlap"""
    target_labels = target_sample[labels].to_numpy().astype(int)
    similarities = []

    for _, ret_sample in retrieval_samples.iterrows():
        ret_labels = ret_sample[labels].to_numpy().astype(int)

        # Jaccard similarity: intersection / union
        intersection = np.sum(target_labels & ret_labels)
        union = np.sum(target_labels | ret_labels)

        similarity = intersection / union if union > 0 else 0.0
        similarities.append(similarity)

    return np.array(similarities)


def prepare_minimal_prompt(
        retrieval_samples,
        target_sample,
        target_similarity,
        k,
        prompt_templates,
        filter_type,
        prompt_type,
        section_type,
        labels,
        findings_col,
        impression_col,
        study_id_col,
):
    """Minimal prompt preparation"""
    # Sort by similarity
    sim_sort_idxs = target_similarity.argsort()[::-1]  # Descending order

    # Apply filtering
    if filter_type == "exact":
        target_positives = (target_sample[labels] == 1).astype(int)
        mask = []
        for idx in sim_sort_idxs:
            ret_positives = (retrieval_samples.iloc[idx][labels] == 1).astype(int)
            if (target_positives == ret_positives).all():
                mask.append(idx)
        selected_idxs = mask[:k]
    elif filter_type == "partial":
        target_positives = (target_sample[labels] == 1).astype(int)
        overlaps = []
        for idx in sim_sort_idxs:
            ret_positives = (retrieval_samples.iloc[idx][labels] == 1).astype(int)
            overlap = np.sum(target_positives & ret_positives)
            overlaps.append((idx, overlap))
        # Sort by overlap count
        overlaps.sort(key=lambda x: x[1], reverse=True)
        selected_idxs = [x[0] for x in overlaps[:k]]
    else:  # no-filter
        selected_idxs = sim_sort_idxs[:k]

    # Select retrieval samples
    k_references = retrieval_samples.iloc[selected_idxs]

    # Build examples
    examples = []
    retrieved_studies = []

    for i, (_, reference) in enumerate(k_references.iterrows()):
        example = f"Example: {i + 1}\n"

        # Add labels
        if prompt_type != "naive":
            pos_labels = reference[labels][reference[labels] == 1].index.tolist()
            if prompt_type == "simple":
                example += f"Labels: {', '.join(pos_labels)}\n"
            else:  # verbose/instruct
                example += f"Positive: {', '.join(pos_labels)}\n"

        # Add report content
        if section_type in ["findings", "both"]:
            example += f"{reference[findings_col]}\n"
        if section_type in ["impression", "both"]:
            example += f"{reference[impression_col]}\n"

        examples.append(example)
        retrieved_studies.append(str(reference[study_id_col]))

    # Build final prompt
    context = "\n".join(examples)
    template = prompt_templates[prompt_type]

    # Add target labels
    if prompt_type != "naive":
        target_pos_labels = target_sample[labels][target_sample[labels] == 1].index.tolist()
        if prompt_type == "simple":
            target_label_text = f"Labels: {', '.join(target_pos_labels)}\n"
        else:
            target_label_text = f"Positive: {', '.join(target_pos_labels)}\n"
        prompt = template.format(context, target_label_text)
    else:
        prompt = template.format(context, "")

    # Target report (for evaluation)
    target_report = ""
    if section_type in ["findings", "both"]:
        target_report += target_sample[findings_col] + "\n"
    if section_type in ["impression", "both"]:
        target_report += target_sample[impression_col] + "\n"

    return prompt, target_report, ", ".join(retrieved_studies)


def save_results(results, output_path, header=True):
    """Save results to CSV"""
    df = pd.DataFrame(results)
    df.to_csv(output_path, mode='a' if not header else 'w',
              header=header, index=False)


def main():
    parser = argparse.ArgumentParser(description="Minimal Radiology Report Generator")
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--reports_csv", required=True, help="Reports CSV file")
    parser.add_argument("--labels_csv", required=True, help="Labels CSV file")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieval samples")
    parser.add_argument("--filter_type", choices=FILTER_TYPE, default="no-filter")
    parser.add_argument("--prompt_type", choices=PROMPT_TYPE, default="simple")
    parser.add_argument("--section_type", choices=SECTION_TYPE, default="findings")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--prompt_yaml", default="./rrg/prompts_en.yaml")
    parser.add_argument("--output_dir", default="./output")

    args = parser.parse_args()

    generate_from_labels_only(
        model=args.model,
        reports_csv=args.reports_csv,
        labels_csv=args.labels_csv,
        k=args.k,
        filter_type=args.filter_type,
        prompt_type=args.prompt_type,
        section_type=args.section_type,
        batch_size=args.batch_size,
        prompt_yaml=args.prompt_yaml,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()