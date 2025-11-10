"""
Problem-specific utilities loader.

Dynamically imports problem-specific visualization and evaluation utilities.
"""


def get_visualization_module(problem_name: str):
    """
    Dynamically import visualization module for the problem.
    
    Args:
        problem_name: Name of the problem (e.g., 'schrodinger', 'problem2')
        
    Returns:
        Tuple of (visualize_dataset, visualize_evaluation) functions
        
    Raises:
        ValueError: If problem name is unknown
    """
    if problem_name == 'schrodinger':
        from .schrodinger_viz import visualize_dataset, visualize_evaluation
        return visualize_dataset, visualize_evaluation
    elif problem_name == 'problem2':
        from .problem2_viz import visualize_dataset, visualize_evaluation
        return visualize_dataset, visualize_evaluation
    else:
        raise ValueError(f"Unknown problem: {problem_name}")

