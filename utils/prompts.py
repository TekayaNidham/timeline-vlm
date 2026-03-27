"""
Prompt templates for time probing experiments.

Section 5.1 / Table 2 of the paper: 9 prompt formulations (P1-P9)
that explore different ways of referencing the first appearance of an object.
"""


def get_prompt_templates():
    """
    Get all 9 prompt templates for time probing (Table 2).

    Returns:
        Dictionary mapping prompt IDs (P1-P9) to template strings.
        Templates use {year} as placeholder.
    """
    return {
        'P1': '{year}',
        'P2': 'year {year}',
        'P3': 'was released in the year {year}',
        'P4': 'was invented in the year {year}',
        'P5': 'was first introduced in the year {year}',
        'P6': 'was created in the year {year}',
        'P7': 'was built in the year {year}',       # Best performing
        'P8': 'first appeared in the year {year}',
        'P9': 'emerged in the year {year}',
    }


def get_best_prompts_by_model(model_name):
    """
    Get recommended prompts for specific models based on paper results.

    P7 is consistently best across all models (Table 2).
    """
    model_name = model_name.lower()
    if 'eva' in model_name:
        return ['P7', 'P5', 'P4']
    elif 'clip' in model_name:
        return ['P7', 'P8', 'P3']
    else:
        return ['P7', 'P8', 'P5']


def format_prompt(template, year):
    """Format a prompt template with a specific year."""
    return template.format(year=year)


# Paper results for prompt sensitivity analysis (Table 2)
PROMPT_RESULTS = {
    'clip': {
        'P1': {'mae': 48.28, 'tai': 0.70},
        'P2': {'mae': 19.89, 'tai': 0.82},
        'P3': {'mae': 17.86, 'tai': 0.83},
        'P4': {'mae': 19.34, 'tai': 0.81},
        'P5': {'mae': 15.84, 'tai': 0.82},
        'P6': {'mae': 19.06, 'tai': 0.77},
        'P7': {'mae': 8.79, 'tai': 0.86},
        'P8': {'mae': 15.19, 'tai': 0.82},
        'P9': {'mae': 17.87, 'tai': 0.80},
    },
    'eva-clip': {
        'P1': {'mae': 48.14, 'tai': 0.73},
        'P2': {'mae': 16.34, 'tai': 0.84},
        'P3': {'mae': 11.30, 'tai': 0.88},
        'P4': {'mae': 9.18, 'tai': 0.89},
        'P5': {'mae': 7.70, 'tai': 0.89},
        'P6': {'mae': 10.30, 'tai': 0.88},
        'P7': {'mae': 7.44, 'tai': 0.89},
        'P8': {'mae': 14.24, 'tai': 0.85},
        'P9': {'mae': 14.81, 'tai': 0.86},
    },
}
