"""
Create Submission Script

This script converts detected circuit JSON files from the 'detected_paths' directory
into submission-ready JSON files in the 'submissions' directory with proper naming convention.
"""

import os
import json
import glob
from datetime import datetime
from path_to_graph_translator import convert_json_file_to_leaderboard_format


def extract_model_task_from_filename(filename: str) -> tuple:
    """
    Extract model and task from filename.
    Expected format: detected_circuit_{model}_{task}_{timestamp}.json
    
    Args:
        filename: The filename to parse
        
    Returns:
        Tuple of (model, task)
    """
    # Remove .json extension and split by underscores
    parts = filename.replace('.json', '').split('_')
    
    if len(parts) >= 4:
        # detected_circuit_{model}_{task}_{timestamp}
        model = parts[2]  # e.g., 'gpt2'
        task = parts[3]   # e.g., 'ioi'
        return model, task
    else:
        # Fallback for unexpected formats
        return 'unknown_model', 'unknown_task'


def create_submission_directory() -> str:
    """
    Create submission directory with timestamp.
    
    Returns:
        Path to the created submission directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir = os.path.join("submissions", f"submission_{timestamp}") 
    os.makedirs(submission_dir, exist_ok=True)
    return submission_dir


def process_detected_paths(input_dir: str = "detected_paths", output_dir: str = None):
    """
    Process all detected circuit JSON files and convert them to submission format.
    
    Args:
        input_dir: Directory containing detected circuit JSON files
        output_dir: Directory to save submission files (if None, creates timestamped directory)
    """
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = create_submission_directory()
    else:
        os.makedirs(os.path.join("submissions", output_dir), exist_ok=True)
    
    print(f"Processing files from '{input_dir}' directory...")
    print(f"Output will be saved to '{output_dir}' directory")
    print("=" * 60)
    
    # Find all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{input_dir}' directory!")
        return
    
    print(f"Found {len(json_files)} JSON files to process:")
    for file in json_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each file
    processed_files = []
    
    for json_file in json_files:
        filename = os.path.basename(json_file)
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        
        try:
            # Extract model and task
            model, task = extract_model_task_from_filename(filename)
            print(f"Extracted: model='{model}', task='{task}'")
            
            # Create task/model directory (required by submission format)
            task_model_dir = os.path.join(output_dir, f"{task}_{model}")
            os.makedirs(task_model_dir, exist_ok=True)
            
            # Convert to submission format
            print("Converting to submission format...")
            graph_data = convert_json_file_to_leaderboard_format(
                input_filename=json_file,
                output_filename=os.path.join(task_model_dir, f"{task}_{model}.json"),
                min_score_threshold=0.0,
                normalize_scores=True
            )
            
            print(f"✓ Created submission file: {task_model_dir}/{task}_{model}.json")
            print(f"  - Nodes: {len(graph_data['nodes'])}")
            print(f"  - Edges: {len(graph_data['edges'])}")
            
            processed_files.append((model, task, task_model_dir))
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed {len(processed_files)} files:")
    
    for model, task, task_model_dir in processed_files:
        print(f"  - {task}_{model}: {task_model_dir}")
    
    print(f"\nSubmission directory: {output_dir}")
    print("\nNext steps:")
    print("1. Review the generated files")
    print("2. Upload the submission directory to HuggingFace")
    print("3. Submit the HuggingFace URL to the MIB leaderboard")
    
    return output_dir


def main():
    """
    Main function to run the submission creation process.
    """
    print("MIB Circuit Submission Creator")
    print("=" * 50)
    print("This script converts detected circuit JSON files to submission format.")
    print("Expected directory structure:")
    print("  detected_paths/")
    print("    ├── detected_circuit_gpt2_ioi_20250722_104554.json")
    print("    ├── detected_circuit_qwen_mcqa_20250722_104555.json")
    print("    └── ...")
    print()
    
    # Check if detected_paths directory exists
    if not os.path.exists("detected_paths"):
        print("❌ 'detected_paths' directory not found!")
        print("Please create the directory and add your detected circuit JSON files.")
        return
    
    # Process all files
    submission_dir = process_detected_paths()
    
    if submission_dir:
        print(f"\n🎉 Submission files created successfully in: {submission_dir}")


if __name__ == "__main__":
    main() 