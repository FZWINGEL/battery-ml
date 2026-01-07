"""Experiment configuration helper for LG M50T dataset.

Provides experiment-specific cell IDs and temperature mappings.
"""

from typing import Dict, List, Any


def get_experiment_config(experiment_id: int) -> Dict[str, Any]:
    """Get cell IDs and temperature mapping for an experiment.
    
    Based on Dataset.md specifications for LG M50T experiments.
    
    Args:
        experiment_id: Experiment ID (1-5)
    
    Returns:
        Dict with 'cells' (list of cell IDs) and 'temp_map' (dict mapping temp to cell IDs)
    
    Raises:
        ValueError: If experiment_id is not in range 1-5
    """
    configs = {
        1: {
            'cells': ['A', 'B', 'D', 'E', 'F', 'J', 'K', 'L', 'M'],
            'temp_map': {
                10: ['A', 'B', 'J'],
                25: ['D', 'E', 'F'],
                40: ['K', 'L', 'M']
            }
        },
        2: {
            'cells': ['A', 'B', 'C', 'D', 'E', 'F'],
            'temp_map': {
                10: ['A', 'B'],
                25: ['C', 'D'],
                40: ['E', 'F']
            }
        },
        3: {
            'cells': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            'temp_map': {
                10: ['A', 'B', 'C'],
                25: ['D', 'E', 'F'],
                40: ['G', 'H', 'I']
            }
        },
        4: {
            'cells': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            'temp_map': {
                10: ['A', 'B', 'C'],
                25: ['D', 'E'],
                40: ['F', 'G', 'H']
            }
        },
        5: {
            'cells': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            'temp_map': {
                10: ['A', 'B', 'C'],
                25: ['D', 'E'],
                40: ['F', 'G', 'H']
            }
        }
    }
    
    if experiment_id not in configs:
        raise ValueError(f"Experiment ID must be 1-5, got {experiment_id}")
    
    return configs[experiment_id]


def get_experiment_name(experiment_id: int) -> str:
    """Get human-readable experiment name.
    
    Args:
        experiment_id: Experiment ID (1-5)
    
    Returns:
        Experiment name string
    """
    names = {
        1: "Si-based Degradation",
        2: "C-based Degradation 2",
        3: "Cathode Degradation and Li-Plating",
        4: "Drive Cycle Aging (Control)",
        5: "Standard Cycle Aging (Control)"
    }
    return names.get(experiment_id, f"Experiment {experiment_id}")
