import json
import os
from collections import defaultdict

def analyze_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_records = len(data)
    explanation_changed_count = 0
    score7_changed_count = 0
    total_time_spent = 0
    total_initial_errors = 0
    
    # Log action counts
    log_action_counts = defaultdict(int)

    for key, record in data.items():
        # 1. Explanation changes
        if record.get("explanation") != record.get("after_explanation"):
            explanation_changed_count += 1
        
        # 2. Score7 changes
        if record.get("score7") != record.get("after_score7"):
            score7_changed_count += 1
            
        # 3. Time spent
        total_time_spent += record.get("time_spent", 0)
        
        # 4. Total Initial Errors
        errors = record.get("errors", [])
        total_initial_errors += len(errors)
        
        # 5. Log Action Analysis
        logs = record.get("logs", [])
        for log_entry in logs:
            action = log_entry.get("action")
            if action:
                log_action_counts[action] += 1
        
        # 6. Diff Analysis for No-edit and Severity
        after_errors = record.get("after_errors", [])
        for e1 in errors:
            # Try to find exact positional match in after_errors
            match = None
            for e2 in after_errors:
                if e1['start'] == e2['start'] and e1['end'] == e2['end']:
                    match = e2
                    break
            
            if match:
                if e1.get('severity') == match.get('severity'):
                    log_action_counts["No-edit"] += 1
                else:
                    log_action_counts["Severity"] += 1

    # Output Results
    print(f"Total Records: {total_records}")
    print(f"Total Initial Errors: {total_initial_errors}")
    print(f"Explanation Changed: {explanation_changed_count} ({explanation_changed_count/total_records:.2%})")
    print(f"Score7 Changed: {score7_changed_count} ({score7_changed_count/total_records:.2%})")
    
    avg_time = total_time_spent / total_records if total_records > 0 else 0
    print(f"Average Time Spent: {avg_time:.2f} seconds")
    
    print("\nAction Counts (Percentage relative to Total Initial Errors):")
    
    # Ensure all expected keys are present for display
    expected_actions = ["No-edit", "Add", "Remove", "Resize", "Move", "Reset", "Severity"]
    for action in expected_actions:
        if action not in log_action_counts:
            log_action_counts[action] = 0
            
    sorted_actions = sorted(log_action_counts.items(), key=lambda x: x[1], reverse=True)
    
    for action, count in sorted_actions:
        percentage = (count / total_initial_errors * 100) if total_initial_errors > 0 else 0
        print(f"{action}: {count} ({percentage:.2f}%)")


if __name__ == "__main__":
    file_path = "gemini-3_after_human_annotation.json"
    if os.path.exists(file_path):
        analyze_data(file_path)
    else:
        print(f"File not found: {file_path}")
