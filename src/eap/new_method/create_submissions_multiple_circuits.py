"""
Create Submission Script for Multiple Circuits (.pt format)

This script converts detected circuit JSON files from the 'detected_paths' directory
into submission-ready .pt files for various circuit sizes, organized by task and model.
"""

import os
import glob
from datetime import datetime
from path_to_graph_translator import convert_json_to_pt_submissions, extract_model_task_from_filename

# Percentages of edges to include in the circuits
# Corresponds to {0%, 0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 20%, 50%}
CIRCUIT_PERCENTAGES = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]


def create_submission_directory() -> str:
    """
    Create a timestamped submission directory.
    
    Returns:
        Path to the created submission directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir = os.path.join("submissions_pt", f"submission_{timestamp}") 
    os.makedirs(submission_dir, exist_ok=True)
    return submission_dir


def process_detected_paths_to_pt(input_dir: str = "detected_paths", output_dir: str = None):
    """
    Process detected circuit JSONs and convert them to .pt submission format for multiple percentages.
    
    Args:
        input_dir: Directory with detected circuit JSON files.
        output_dir: Directory to save submission files. If None, creates a new timestamped directory.
    """
    
    if output_dir is None:
        output_dir = create_submission_directory()
    else:
        # Allows specifying a custom output directory name
        output_dir = os.path.join("submissions_pt", output_dir)
        os.makedirs(output_dir, exist_ok=True)

    print(f"Processing files from '{input_dir}' directory...")
    print(f"Output will be saved to '{output_dir}' directory")
    print(f"Generating circuits for percentages: {CIRCUIT_PERCENTAGES}")
    print("=" * 70)
    
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{input_dir}'.")
        return
        
    print(f"Found {len(json_files)} files to process:")
    for file in json_files:
        print(f"  - {os.path.basename(file)}")

    processed_files = []
    
    for json_file in json_files:
        filename = os.path.basename(json_file)
        print(f"\n{'='*70}")
        print(f"Processing: {filename}")
        print(f"{'='*70}")
        
        try:
            model, task = extract_model_task_from_filename(filename)
            print(f"Extracted: model='{model}', task='{task}'")
            
            task_model_dir = os.path.join(output_dir, f"{task}_{model}")
            os.makedirs(task_model_dir, exist_ok=True)
            
            print(f"Converting to .pt format for {len(CIRCUIT_PERCENTAGES)} percentages...")
            convert_json_to_pt_submissions(
                input_filename=json_file,
                output_dir=task_model_dir,
                percentages=CIRCUIT_PERCENTAGES
            )
            
            print(f"✓ Created .pt submission files in: {task_model_dir}")
            processed_files.append((model, task, task_model_dir))
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
            
    # Summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Successfully processed {len(processed_files)} files:")
    
    for model, task, task_model_dir in processed_files:
        print(f"  - {task}_{model}: {task_model_dir}")
        
    print(f"\nSubmission directory: {output_dir}")
    print("\nNext steps:")
    print("1. Review the generated .pt files in each folder.")
    print("2. Upload the submission directory to the competition platform.")


def main():
    """
    Main function to run the submission creation process.
    """
    print("MIB Circuit Submission Creator (.pt format)")
    print("=" * 50)
    print("This script converts detected circuit JSONs to .pt submission files for multiple thresholds.")
    
    if not os.path.exists("detected_paths"):
        print("❌ 'detected_paths' directory not found!")
        print("Please ensure it exists and contains your circuit JSON files.")
        return
        
    process_detected_paths_to_pt()


if __name__ == "__main__":
    main() 