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
        Tuple of visualization functions:
        (visualize_dataset, visualize_evaluation, visualize_ncc_dataset, 
         visualize_ncc_classification, visualize_ncc_classification_input_space,
         visualize_ncc_classification_heatmap, visualize_ncc_classification_input_space_heatmap)
        
    Raises:
        ValueError: If problem name is unknown
    """
    if problem_name == 'schrodinger':
        from .schrodinger_viz import (
            visualize_dataset, visualize_evaluation, 
            visualize_ncc_dataset, visualize_ncc_classification,
            visualize_ncc_classification_input_space,
            visualize_ncc_classification_heatmap,
            visualize_ncc_classification_input_space_heatmap
        )
        return (visualize_dataset, visualize_evaluation, visualize_ncc_dataset, 
                visualize_ncc_classification, visualize_ncc_classification_input_space,
                visualize_ncc_classification_heatmap, visualize_ncc_classification_input_space_heatmap)
    elif problem_name == 'problem2':
        from .problem2_viz import (visualize_dataset, visualize_evaluation, 
                                   visualize_ncc_dataset, visualize_ncc_classification,
                                   visualize_ncc_classification_input_space)
        # problem2 doesn't have heatmap functions yet, return None for those
        return (visualize_dataset, visualize_evaluation, visualize_ncc_dataset, 
                visualize_ncc_classification, visualize_ncc_classification_input_space,
                None, None)
    else:
        raise ValueError(f"Unknown problem: {problem_name}")

