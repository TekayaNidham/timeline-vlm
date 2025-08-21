
def get_prompt_templates():
    """
    Get all prompt templates for time probing.
    
    Returns:
        Dictionary mapping prompt IDs to template strings
    """
    prompts = {
        'P1': '{year}',  # Minimalistic - just the year
        'P2': 'year {year}',  # Minimalistic with 'year' prefix
        'P3': 'was released in the year {year}',
        'P4': 'was invented in the year {year}',
        'P5': 'was first introduced in the year {year}',
        'P6': 'was created in the year {year}',
        'P7': 'was built in the year {year}',  # Best performing in paper
        'P8': 'first appeared in the year {year}',
        'P9': 'emerged in the year {year}'
    }
    
    return prompts


def get_best_prompts_by_model(model_name):
    """
    Get recommended prompts for specific models based on paper results.
    
    Args:
        model_name: Name of the model
        
    Returns:
        List of recommended prompt IDs
    """
    model_name = model_name.lower()
    
    # Based on results from the paper
    if 'eva' in model_name:
        # EVA-CLIP performs well with P7, P5, P4
        return ['P7', 'P5', 'P4']
    elif 'clip' in model_name:
        # Standard CLIP performs best with P7
        return ['P7', 'P8', 'P3']
    else:
        # Default recommendations
        return ['P7', 'P8', 'P5']


def format_prompt(template, year):
    """
    Format a prompt template with a specific year.
    
    Args:
        template: Prompt template string
        year: Year to insert
        
    Returns:
        Formatted prompt string
    """
    return template.format(year=year)


def get_prompt_analysis():
    """
    Get analysis of prompt performance from the paper.
    
    Returns:
        Dictionary with prompt analysis results
    """
    analysis = {
        'clip': {
            'P1': {'mae': 48.28, 'tai': 0.70},  # Poor - too minimal
            'P2': {'mae': 19.89, 'tai': 0.82},  # Better but still minimal
            'P3': {'mae': 17.86, 'tai': 0.83},  # Good
            'P4': {'mae': 19.34, 'tai': 0.81},  # Moderate
            'P5': {'mae': 15.84, 'tai': 0.82},  # Good
            'P6': {'mae': 19.06, 'tai': 0.77},  # Moderate
            'P7': {'mae': 8.79, 'tai': 0.86},   # Best!
            'P8': {'mae': 15.19, 'tai': 0.82},  # Good
            'P9': {'mae': 17.87, 'tai': 0.80}   # Moderate
        },
        'eva-clip': {
            'P1': {'mae': 48.14, 'tai': 0.73},  # Poor - too minimal
            'P2': {'mae': 16.34, 'tai': 0.84},  # Moderate
            'P3': {'mae': 11.30, 'tai': 0.88},  # Good
            'P4': {'mae': 9.18, 'tai': 0.89},   # Very good
            'P5': {'mae': 7.70, 'tai': 0.89},   # Very good
            'P6': {'mae': 10.30, 'tai': 0.88},  # Good
            'P7': {'mae': 7.44, 'tai': 0.89},   # Best!
            'P8': {'mae': 14.24, 'tai': 0.85},  # Good
            'P9': {'mae': 14.81, 'tai': 0.86}   # Good
        }
    }
    
    return analysis


def print_prompt_recommendations(model_name=None):
    """
    Print prompt recommendations and analysis.
    
    Args:
        model_name: Optional model name for specific recommendations
    """
    prompts = get_prompt_templates()
    analysis = get_prompt_analysis()
    
    print("Time Probing Prompt Templates:")
    print("=" * 60)
    
    for pid, template in prompts.items():
        example = format_prompt(template, 2020)
        print(f"{pid}: \"{example}\"")
    
    print("\n" + "=" * 60)
    print("Performance Analysis (from paper):")
    print("=" * 60)
    
    # Show performance table
    print("\nCLIP Performance:")
    print(f"{'Prompt':<8} {'Template':<40} {'MAE':<8} {'TAI':<8}")
    print("-" * 64)
    for pid in sorted(prompts.keys()):
        template = prompts[pid]
        if pid in analysis['clip']:
            mae = analysis['clip'][pid]['mae']
            tai = analysis['clip'][pid]['tai']
            marker = " *" if pid == 'P7' else ""
            print(f"{pid:<8} {template:<40} {mae:<8.2f} {tai:<8.3f}{marker}")
    
    print("\nEVA-CLIP Performance:")
    print(f"{'Prompt':<8} {'Template':<40} {'MAE':<8} {'TAI':<8}")
    print("-" * 64)
    for pid in sorted(prompts.keys()):
        template = prompts[pid]
        if pid in analysis['eva-clip']:
            mae = analysis['eva-clip'][pid]['mae']
            tai = analysis['eva-clip'][pid]['tai']
            marker = " *" if pid == 'P7' else ""
            print(f"{pid:<8} {template:<40} {mae:<8.2f} {tai:<8.3f}{marker}")
    
    print("\n* = Best performing prompt")
    
    if model_name:
        recommended = get_best_prompts_by_model(model_name)
        print(f"\nRecommended prompts for {model_name}: {', '.join(recommended)}")
    
    print("\nKey findings:")
    print("- Minimalistic prompts (P1, P2) perform poorly")
    print("- Descriptive prompts with context perform better")
    print("- 'was built...' (P7) consistently performs best")
    print("- EVA-CLIP is less sensitive to prompt variations than CLIP")


if __name__ == "__main__":
    # Example usage
    print_prompt_recommendations()